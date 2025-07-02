import Foundation
import LocalLLMClient
import Jinja

final class Model {
    let model: OpaquePointer
    let chatTemplate: String

    var vocab: OpaquePointer {
        llama_model_get_vocab(model)
    }

    init(url: URL) throws(LLMError) {
        // Validate file before attempting to load
        let fileManager = FileManager.default
        
        
        guard fileManager.fileExists(atPath: url.path) else {
            throw LLMError.failedToLoad(reason: "Model file does not exist at path: \(url.path)")
        }
        
        // Check if it's actually a file (not a directory)
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            throw LLMError.failedToLoad(reason: "Path exists but is not a file: \(url.path)")
        }
        
        // Check file accessibility
        guard fileManager.isReadableFile(atPath: url.path) else {
            throw LLMError.failedToLoad(reason: "Model file is not readable: \(url.path)")
        }
        
        // Get file size for validation
        do {
            let attributes = try fileManager.attributesOfItem(atPath: url.path)
            let fileSize = attributes[.size] as? Int64 ?? 0
            
            if fileSize == 0 {
                throw LLMError.failedToLoad(reason: "Model file is empty: \(url.path)")
            }
            
        } catch {
            throw LLMError.failedToLoad(reason: "Cannot read model file attributes: \(error.localizedDescription)")
        }
        
        // Check if file appears to be a GGUF file (basic validation)
        do {
            let fileHandle = try FileHandle(forReadingFrom: url)
            defer { fileHandle.closeFile() }
            
            let header = fileHandle.readData(ofLength: 4)
            if header.count >= 4 {
                let magic = String(data: header, encoding: .ascii) ?? ""
                // GGUF files should start with "GGUF"
                if !magic.hasPrefix("GGUF") && !magic.hasPrefix("GGM") && isVerboseLoggingEnabled() {
                    print("[LLAMA MODEL] Warning: File does not appear to be a GGUF model (header: \(magic.debugDescription))")
                }
            }
        } catch {
        }

        var model_params = llama_model_default_params()
#if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
#endif
        model_params.use_mmap = true


        guard let model = llama_model_load_from_file(url.path(), model_params) else {
            throw LLMError.failedToLoad(reason: "llama_model_load_from_file failed for: \(url.path) - check llama.cpp logs above for details")
        }
        

        self.model = model

        let chatTemplate = getString(capacity: 8192) { buffer, length in
            // LLM_KV_TOKENIZER_CHAT_TEMPLATE
            llama_model_meta_val_str(model, "tokenizer.chat_template", buffer, length)
        }

        // If the template is empty, it uses Gemma3-styled template as default
        self.chatTemplate = chatTemplate.isEmpty ? #"{{ bos_token }} {%- if messages[0]['role'] == 'system' -%} {%- if messages[0]['content'] is string -%} {%- set first_user_prefix = messages[0]['content'] + ' ' -%} {%- else -%} {%- set first_user_prefix = messages[0]['content'][0]['text'] + ' ' -%} {%- endif -%} {%- set loop_messages = messages[1:] -%} {%- else -%} {%- set first_user_prefix = "" -%} {%- set loop_messages = messages -%} {%- endif -%} {%- for message in loop_messages -%} {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%} {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }} {%- endif -%} {%- if (message['role'] == 'assistant') -%} {%- set role = "model" -%} {%- else -%} {%- set role = message['role'] -%} {%- endif -%} {{ (first_user_prefix if loop.first else "") }} {%- if message['content'] is string -%} {{ message['content'] | trim }} {%- elif message['content'] is iterable -%} {%- for item in message['content'] -%} {%- if item['type'] == 'image' -%} {{ '<start_of_image>' }} {%- elif item['type'] == 'text' -%} {{ item['text'] | trim }} {%- endif -%} {%- endfor -%} {%- else -%} {{ raise_exception("Invalid content type") }} {%- endif -%} {%- endfor -%}"# : chatTemplate
    }

    deinit {
        llama_model_free(model)
    }

    func makeAndAllocateContext(with ctx_params: llama_context_params) throws(LLMError) -> OpaquePointer {
        
        guard let context = llama_init_from_model(model, ctx_params) else {
            // Enhanced error message with specific failure details
            let modelCtxSize = llama_n_ctx_train(model)
            let requestedCtx = ctx_params.n_ctx
            let availableMemory = ProcessInfo.processInfo.physicalMemory
            
            var failureReasons: [String] = []
            
            if requestedCtx > modelCtxSize {
                failureReasons.append("Requested context size (\(requestedCtx)) exceeds model maximum (\(modelCtxSize))")
            }
            
            if ctx_params.n_threads <= 0 {
                failureReasons.append("Invalid thread count (\(ctx_params.n_threads))")
            }
            
            // Estimate memory requirements (rough calculation)
            let estimatedMemoryMB = (Int(requestedCtx) * Int(llama_model_n_embd(model)) * 4) / (1024 * 1024)
            if estimatedMemoryMB > availableMemory / (1024 * 1024) {
                failureReasons.append("Estimated memory requirement (\(estimatedMemoryMB)MB) may exceed available memory")
            }
            
            let reasonString = failureReasons.isEmpty ? 
                "Unknown context creation failure - check llama.cpp logs above for details" :
                failureReasons.joined(separator: "; ")
            
            
            throw LLMError.invalidParameter(reason: "Failed to create context: \(reasonString)")
        }
        
        
        return context
    }

    func tokenizerConfigs() -> [String: Any] {
        let numberOfConfigs = llama_model_meta_count(model)
        return (0..<numberOfConfigs).reduce(into: [:]) { partialResult, i in
            let key = getString(capacity: 64) { buffer, length in
                llama_model_meta_key_by_index(model, i, buffer, length)
            }
            let value = getString(capacity: 2048) { buffer, length in
                llama_model_meta_val_str_by_index(model, i, buffer, length)
            }
            partialResult[key] = value
        }
    }
}

private func getString(capacity: Int = 1024, getter: (UnsafeMutablePointer<CChar>?, Int) -> Int32) -> String {
    String(unsafeUninitializedCapacity: capacity) { buffer in
        buffer.withMemoryRebound(to: CChar.self) { buffer in
            let length = Int(getter(buffer.baseAddress, capacity))
            return max(0, length)
        }
    }
}
