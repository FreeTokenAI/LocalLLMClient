#if BUILD_DOCC
@preconcurrency @_implementationOnly import llama
#elseif canImport(llama)
@preconcurrency private import llama
#else
@preconcurrency import LocalLLMClientLlamaC
#endif
import Foundation
import LocalLLMClient

public final class Context: @unchecked Sendable {
    let parameter: LlamaClient.Parameter
    package let context: OpaquePointer
    package var batch: llama_batch
    var sampling: Sampler
    let grammer: Sampler?
    let cursorPointer: UnsafeMutableBufferPointer<llama_token_data>
    let model: Model
    let extraEOSTokens: Set<String>
    private var promptCaches: [(chunk: MessageChunk, lastPosition: llama_pos)] = []

    package var vocab: OpaquePointer {
        model.vocab
    }

    package var numberOfBatch: Int32 {
        Int32(llama_n_batch(context))
    }

    package var position: Int32 {
        guard let kv = llama_get_memory(context) else {
            return -1
        }

        return llama_memory_seq_pos_max(kv, 0) + 1
    }

    public init(url: URL, parameter: LlamaClient.Parameter = .default) throws(LLMError) {
        initializeLlama()
        
        // Enable verbose logging if requested
        if parameter.options.verbose {
            setLlamaVerbose(true)
        }


        // Validate and prepare context parameters
        var ctx_params = llama_context_default_params()
        
        // Validate context size
        if parameter.context <= 0 {
            throw LLMError.invalidParameter(reason: "Context size must be positive, got: \(parameter.context)")
        }
        ctx_params.n_ctx = UInt32(parameter.context)
        
        // Calculate thread count with validation
        let processorCount = ProcessInfo.processInfo.processorCount
        let calculatedThreads = parameter.numberOfThreads ?? max(1, min(8, processorCount - 2))
        
        if calculatedThreads <= 0 {
            throw LLMError.invalidParameter(reason: "Thread count must be positive, calculated: \(calculatedThreads)")
        }
        
        ctx_params.n_threads = Int32(calculatedThreads)
        ctx_params.n_threads_batch = ctx_params.n_threads
        
        // Validate batch size
        if parameter.batch <= 0 {
            throw LLMError.invalidParameter(reason: "Batch size must be positive, got: \(parameter.batch)")
        }


        self.parameter = parameter
        
        // Load model
        self.model = try Model(url: url)
        
        // Create context
        self.context = try model.makeAndAllocateContext(with: ctx_params)
        batch = llama_batch_init(Int32(parameter.batch), 0, 1)
        extraEOSTokens = parameter.options.extraEOSTokens

        // https://github.com/ggml-org/llama.cpp/blob/master/common/sampling.cpp
        sampling = llama_sampler_chain_init(llama_sampler_chain_default_params())
        
        let minKeep = 0
        let penaltyFreq: Float = 0
        let penaltyPresent: Float = 0
        
        // Validate sampler parameters before using them
        if parameter.temperature < 0 {
            throw LLMError.invalidParameter(reason: "Temperature must be non-negative, got: \(parameter.temperature)")
        }
        if parameter.topK < 0 {
            throw LLMError.invalidParameter(reason: "TopK must be non-negative, got: \(parameter.topK)")
        }
        if parameter.topP < 0 || parameter.topP > 1 {
            throw LLMError.invalidParameter(reason: "TopP must be between 0 and 1, got: \(parameter.topP)")
        }
        if parameter.typicalP < 0 || parameter.typicalP > 1 {
            throw LLMError.invalidParameter(reason: "TypicalP must be between 0 and 1, got: \(parameter.typicalP)")
        }
        if parameter.penaltyLastN < 0 {
            throw LLMError.invalidParameter(reason: "PenaltyLastN must be non-negative, got: \(parameter.penaltyLastN)")
        }
        if parameter.penaltyRepeat < 0 {
            throw LLMError.invalidParameter(reason: "PenaltyRepeat must be non-negative, got: \(parameter.penaltyRepeat)")
        }
        
        llama_sampler_chain_add(sampling, llama_sampler_init_temp(parameter.temperature))
        llama_sampler_chain_add(sampling, llama_sampler_init_dist(parameter.seed.map(UInt32.init) ?? LLAMA_DEFAULT_SEED))
        llama_sampler_chain_add(sampling, llama_sampler_init_top_k(Int32(parameter.topK)))
        llama_sampler_chain_add(sampling, llama_sampler_init_top_p(parameter.topP, minKeep))
        llama_sampler_chain_add(sampling, llama_sampler_init_min_p(1 - parameter.topP, 1))
        llama_sampler_chain_add(sampling, llama_sampler_init_typical(parameter.typicalP, minKeep))
        llama_sampler_chain_add(sampling, llama_sampler_init_penalties(Int32(parameter.penaltyLastN), parameter.penaltyRepeat, penaltyFreq, penaltyPresent))

        let modelVocab = model.vocab
        let vocabSize = Int(llama_vocab_n_tokens(modelVocab))
        
        if vocabSize <= 0 {
            throw LLMError.invalidParameter(reason: "Invalid vocabulary size: \(vocabSize)")
        }
        
        cursorPointer = .allocate(capacity: vocabSize)

        if let format = parameter.options.responseFormat {
            switch format {
            case .json:
                do {
                    guard let jsonURL = Bundle.module.url(forResource: "json", withExtension: "gbnf") else {
                        throw LLMError.invalidParameter(reason: "JSON grammar template not found in bundle")
                    }
                    
                    let template = try String(contentsOf: jsonURL, encoding: .utf8)
                    grammer = llama_sampler_init_grammar(model.vocab, template, "root")
                    
                    if grammer == nil {
                        throw LLMError.invalidParameter(reason: "Failed to create JSON grammar sampler")
                    }
                } catch let error as LLMError {
                    throw error
                } catch {
                    throw LLMError.failedToLoad(reason: "Failed to load JSON grammar template: \(error.localizedDescription)")
                }
            case let .grammar(grammar, root):
                grammer = llama_sampler_init_grammar(model.vocab, grammar, root)
                
                if grammer == nil {
                    throw LLMError.invalidParameter(reason: "Failed to create custom grammar sampler")
                }
            }
            
            llama_sampler_chain_add(sampling, grammer)
        } else {
            grammer = nil
        }
        
    }

    deinit {
        cursorPointer.deallocate()
        llama_sampler_free(sampling)
        llama_batch_free(batch)
        llama_free(context)
    }

    public func clear() {
        guard let kv = llama_get_memory(context) else {
            return
        }

        llama_memory_clear(kv, true)
    }
    
    // MARK: - KV Cache Optimization
    
    public func getCurrentTokenCount() -> Int32 {
        guard let kv = llama_get_memory(context) else {
            return 0
        }
        
        let maxPos = llama_memory_seq_pos_max(kv, 0)
        let minPos = llama_memory_seq_pos_min(kv, 0)
        
        if maxPos < 0 || minPos < 0 {
            return 0
        }
        
        return maxPos - minPos + 1
    }
    
    public func getMaxPosition() -> Int32 {
        guard let kv = llama_get_memory(context) else {
            return -1
        }
        
        return llama_memory_seq_pos_max(kv, 0)
    }
    
    public func getMinPosition() -> Int32 {
        guard let kv = llama_get_memory(context) else {
            return -1
        }
        
        return llama_memory_seq_pos_min(kv, 0)
    }
    
    public func removeTokenRange(startPos: Int32, endPos: Int32) -> Bool {
        guard let kv = llama_get_memory(context) else {
            return false
        }
        
        return llama_memory_seq_rm(kv, 0, startPos, endPos)
    }
    
    public func shiftTokens(startPos: Int32, endPos: Int32, delta: Int32) {
        guard let kv = llama_get_memory(context) else {
            return
        }
        
        llama_memory_seq_add(kv, 0, startPos, endPos, delta)
    }
    
    public func optimizeKVCache(
        preserveFromStart: Int32,
        preserveFromEnd: Int32,
        targetUsage: Double = 0.6
    ) -> Bool {
        guard let kv = llama_get_memory(context) else {
            return false
        }
        
        let maxPos = llama_memory_seq_pos_max(kv, 0)
        let minPos = llama_memory_seq_pos_min(kv, 0)
        
        guard maxPos >= 0 && minPos >= 0 else {
            return false
        }
        
        let currentTokens = maxPos - minPos + 1
        let contextSize = Int32(parameter.context)
        let targetTokens = Int32(Double(contextSize) * targetUsage)
        
        // Only optimize if we're using more than 80% of context
        let currentUsage = Double(currentTokens) / Double(contextSize)
        guard currentUsage > 0.8 else {
            return false
        }
        
        // Calculate removal range
        let preserveStartEnd = minPos + preserveFromStart
        let preserveEndStart = maxPos - preserveFromEnd + 1
        
        // Make sure we have something to remove
        guard preserveStartEnd < preserveEndStart else {
            return false
        }
        
        // Remove middle tokens
        let removeSuccess = llama_memory_seq_rm(kv, 0, preserveStartEnd, preserveEndStart)
        guard removeSuccess else {
            return false
        }
        
        // Shift remaining tokens backward to fill gap
        let shiftAmount = preserveStartEnd - preserveEndStart
        llama_memory_seq_add(kv, 0, preserveEndStart, -1, shiftAmount)
        
        return true
    }

    func addCache(for chunk: MessageChunk, position: llama_pos) {
        let endIndex = promptCaches.endIndex - 1
        switch (chunk, promptCaches.last?.chunk) {
        case let (.text(chunkText), .text(cacheText)):
            promptCaches[endIndex] = (chunk: .text(cacheText + chunkText), lastPosition: position)
        case let (.image(chunkImages), .image(cacheImages)):
            promptCaches[endIndex] = (chunk: .image(cacheImages + chunkImages), lastPosition: position)
        case let (.video(chunkVideos), .video(cacheVideos)):
            promptCaches[endIndex] = (chunk: .video(cacheVideos + chunkVideos), lastPosition: position)
        default:
            promptCaches.append((chunk: chunk, lastPosition: position))
        }
    }

    func removeCachedChunks(_ chunks: inout [MessageChunk]) {
        guard let (lastCacheIndex, newChunk) = lastCacheIndex(of: chunks) else {
            return
        }
        chunks = Array(chunks[(lastCacheIndex + 1)...])
        if let newChunk {
            chunks.append(newChunk)
        }
        if promptCaches[lastCacheIndex].lastPosition < position,
           let kv = llama_get_memory(context) {
            assert(llama_memory_seq_rm(kv, 0, promptCaches[lastCacheIndex].lastPosition, position))
        }
        if promptCaches.count > lastCacheIndex {
            promptCaches.removeSubrange((lastCacheIndex + 1)...)
        }
    }

    func lastCacheIndex(of chunks: [MessageChunk]) -> (index: Int, remaining: MessageChunk?)? {
        for (index, (chunk, cache)) in zip(chunks, promptCaches).enumerated() {
            switch (chunk, cache.chunk) {
            case let (.text(chunkText), .text(cacheText)) where chunkText.hasPrefix(cacheText):
                if chunkText == cacheText {
                    return (index, nil)
                } else {
                    return (index, .text(String(chunkText.dropFirst(cacheText.count))))
                }
            case let (.image(chunkImages), .image(cacheImages)) where chunkImages == cacheImages:
                return (index, nil)
            case let (.video(chunkVideos), .video(cacheVideos)) where chunkVideos == cacheVideos:
                return (index, nil)
            default:
                break
            }
        }
        return nil
    }
}
