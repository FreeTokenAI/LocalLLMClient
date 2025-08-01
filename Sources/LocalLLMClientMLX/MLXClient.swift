import LocalLLMClient
import MLX
import MLXLMCommon
import Foundation

/// A client for interacting with MLX models.
///
/// This actor-based class provides methods for generating text streams from various inputs,
/// and handles the communication with the underlying MLX model via the `MLX` and `MLXLMCommon` frameworks.
public final actor MLXClient: LLMClient {
    private let context: Context
    private let parameter: MLXClient.Parameter

    /// Initializes a new MLX client.
    ///
    /// - Parameters:
    ///   - url: The URL of the MLX model directory. This directory should contain the model weights, tokenizer configuration, and any other necessary model files.
    ///   - parameter: The parameters for the MLX model. Defaults to `.default`.
    /// - Throws: An error if the client fails to initialize, for example, if the model files cannot be loaded.
    nonisolated public init(url: URL, parameter: Parameter = .default) async throws {
        context = try await Context(url: url, parameter: parameter)
        self.parameter = parameter
    }

    /// Generates a text stream from the given input.
    ///
    /// This function processes the input, whether it's plain text, a chat template, or structured chat messages,
    /// and prepares it for the MLX model. It then generates text asynchronously.
    ///
    /// - Parameter input: The input to generate text from. This can be plain text, a chat template, or an array of chat messages.
    /// - Returns: An `AsyncStream<String>` that yields text chunks as they are generated by the model.
    /// - Throws: An `LLMError.visionUnsupported` error if the input contains images and the loaded model does not support vision.
    ///           It can also throw errors related to model processing or input preparation.
    public func textStream(from input: LLMInput) async throws -> AsyncStream<String> {
        let chat: [Chat.Message] = switch input.value {
        case .plain(let text):
            [.user(text)]
        case .chatTemplate(let messages):
            messages.map {
                Chat.Message(
                    role: .init(rawValue: $0.value["role"] as? String ?? "") ?? .user,
                    content: $0.value["content"] as? String ?? "",
                    images: $0.attachments.images
                )
            }
        case .chat(let messages):
            messages.map {
                Chat.Message(
                    role: .init(rawValue: $0.role.rawValue) ?? .user,
                    content: $0.content,
                    images: $0.attachments.images
                )
            }
        }

        var userInput = UserInput(chat: chat, additionalContext: ["enable_thinking": false]) // TODO: public API
        userInput.processing.resize = .init(width: 448, height: 448)

        if chat.contains(where: { !$0.images.isEmpty }), !context.supportsVision {
            throw LLMError.visionUnsupported
        }
        let modelContainer =  context.modelContainer

        return try await modelContainer.perform { [userInput] context in
            let lmInput = try await context.processor.prepare(input: userInput)
            let stream = try MLXLMCommon.generate(
                input: lmInput,
                parameters: parameter.parameters,
                context: context
            )

            return .init { continuation in
                let task = Task {
                    for await generated in stream {
                        continuation.yield(generated.chunk ?? "")
                    }
                    continuation.finish()
                }
                continuation.onTermination = { _ in
                    task.cancel()
                }
            }
        }
    }
}

private extension [LLMAttachment] {
    var images: [UserInput.Image] {
        compactMap {
            switch $0.content {
            case let .image(image):
                return try? UserInput.Image.ciImage(llmInputImageToCIImage(image))
            }
        }
    }
}

public extension LocalLLMClient {
    /// Creates a new MLX client.
    ///
    /// This is a factory method for creating `MLXClient` instances.
    ///
    /// - Parameters:
    ///   - url: The URL of the MLX model directory. This directory should contain the model weights, tokenizer configuration, and any other necessary model files.
    ///   - parameter: The parameters for the MLX model. Defaults to `.default`.
    /// - Returns: A new `MLXClient` instance.
    /// - Throws: An error if the client fails to initialize, for example, if the model files cannot be loaded.
    static func mlx(url: URL, parameter: MLXClient.Parameter = .default) async throws -> MLXClient {
        try await MLXClient(url: url, parameter: parameter)
    }
}
