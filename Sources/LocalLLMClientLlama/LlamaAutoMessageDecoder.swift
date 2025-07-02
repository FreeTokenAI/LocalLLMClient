import LocalLLMClient
import Jinja

enum ChatTemplate {
    case `default`
    case gemma3
    case qwen2_5_VL
    case llama3_2V // llama4
    case phi4

    var decoder: any LlamaChatMessageDecoder {
        switch self {
        case .default: LlamaChatMLMessageDecoder()
        case .gemma3: LlamaCustomMessageDecoder(tokenImageRegex: "<start_of_image>")
        case .qwen2_5_VL: LlamaQwen2VLMessageDecoder()
        case .llama3_2V: LlamaLlama3_2VMessageDecoder()
        case .phi4: LlamaChatMLMessageDecoder()
        }
    }
}

public struct LlamaAutoMessageDecoder: LlamaChatMessageDecoder {
    var chatTemplate: ChatTemplate = .default

    public init(chatTemplate: String) throws(LLMError) {
        let template: Template
        do {
            template = try Template(chatTemplate)
        } catch {
            throw LLMError.invalidParameter(reason: "Failed to parse chat template: \(error.localizedDescription)")
        }

        let contentMarker = "$$TEXT$$"
        let image = LLMInputImage()
        let candidateTemplates: [ChatTemplate] = [.gemma3, .qwen2_5_VL, .llama3_2V, .phi4]
        

        do {
            let messages = [
                LLMInput.Message(role: .user, content: contentMarker, attachments: [.image(image)]),
            ]


            for candidate in candidateTemplates {
                
                let value = candidate.decoder.templateValue(from: messages).map(\.value)
                do {
                    // Pick the template that can extract image chunks
                    let rendered = try template.render(["messages": value])
                    let chunks = try candidate.decoder.extractChunks(prompt: rendered, imageChunks: [[image]])
                    if chunks.hasVisionItems() {
                        self.chatTemplate = candidate
                        return
                    }
                } catch {
                    // Continue to next candidate
                }
            }
            
        }
        do {
            let messages = [
                LLMInput.Message(role: .system, content: contentMarker),
                LLMInput.Message(role: .user, content: contentMarker, attachments: [.image(image)]),
                LLMInput.Message(role: .assistant, content: contentMarker),
            ]
            var maxLength = 0
            var bestCandidate: ChatTemplate? = nil

            for candidate in candidateTemplates {
                
                let value = candidate.decoder.templateValue(from: messages).map(\.value)
                do {
                    // Pick the template that can render more characters
                    let rendered = try template.render(["messages": value])
                    if maxLength <= rendered.count {
                        maxLength = rendered.count
                        bestCandidate = candidate
                        self.chatTemplate = candidate
                    }
                } catch {
                    // Continue to next candidate
                }
            }
            
        } catch {
            throw LLMError.failedToLoad(reason: "Template testing failed: \(error.localizedDescription)")
        }
    }

    public func templateValue(from messages: [LLMInput.Message]) -> [LLMInput.ChatTemplateMessage] {
        chatTemplate.decoder.templateValue(from: messages)
    }

    public func applyTemplate(_ messages: [LLMInput.ChatTemplateMessage], chatTemplate: String, additionalContext: [String: Any]?) throws(LLMError) -> String {
        try self.chatTemplate.decoder.applyTemplate(messages, chatTemplate: chatTemplate, additionalContext: additionalContext)
    }

    public func extractChunks(prompt: String, imageChunks: [[LLMInputImage]]) throws -> [MessageChunk] {
        try chatTemplate.decoder.extractChunks(prompt: prompt, imageChunks: imageChunks)
    }

    public func decode(_ messages: [LLMInput.ChatTemplateMessage], context: Context, multimodal: MultimodalContext?) throws {
        try chatTemplate.decoder.decode(messages, context: context, multimodal: multimodal)
    }
}

private extension [MessageChunk] {
    func hasVisionItems() -> Bool {
        contains { chunk in
            switch chunk {
            case .text: false
            case .image, .video: true
            }
        }
    }
}
