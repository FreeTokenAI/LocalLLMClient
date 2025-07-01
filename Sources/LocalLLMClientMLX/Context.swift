import Foundation
import MLXVLM
import LocalLLMClient
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import Tokenizers
#if canImport(OSLog)
import OSLog
#endif

// Simple logger for MLX operations
private struct MLXContextLogger {
    #if canImport(OSLog)
    private let logger = OSLog(subsystem: "com.github.LocalLLMClient", category: "MLX")
    #endif
    
    func info(_ message: String) {
        #if canImport(OSLog)
        os_log("%{public}@", log: logger, type: .info, message)
        #else
        print("[MLX INFO] \(message)")
        #endif
    }
    
    func fault(_ message: String) {
        #if canImport(OSLog)
        os_log("%{public}@", log: logger, type: .fault, message)
        #else
        print("[MLX FAULT] \(message)")
        #endif
    }
    
    func warning(_ message: String) {
        #if canImport(OSLog)
        os_log("%{public}@", log: logger, type: .default, message)
        #else
        print("[MLX WARNING] \(message)")
        #endif
    }
}

public final class Context: Sendable {
    let modelContainer: ModelContainer
    let supportsVision: Bool

    public init(url: URL, parameter: MLXClient.Parameter) async throws(LLMError) {
        let logger = MLXContextLogger()
        
        if parameter.options.verbose {
            logger.info("MLX: Initializing MLX client context")
        }
        
        initializeMLX()

        MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

        let configuration = ModelConfiguration(directory: url, extraEOSTokens: parameter.options.extraEOSTokens)

        let (model, tokenizer) = try await Self.loadModel(
            url: url, configuration: configuration, verbose: parameter.options.verbose
        )
        let (processor, supportsVision) = Self.makeProcessor(
            url: url, configuration: configuration, tokenizer: tokenizer, verbose: parameter.options.verbose
        )

        let context = ModelContext(
            configuration: configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer
        )
        modelContainer = ModelContainer(context: context)
        self.supportsVision = supportsVision
        
        if parameter.options.verbose {
            logger.info("MLX: Context initialization completed successfully - Vision support: \(supportsVision)")
        }
    }

    private static func loadModel(
        url: URL, configuration: ModelConfiguration, verbose: Bool = false
    ) async throws(LLMError) -> (any LanguageModel, any Tokenizer) {
        let logger = MLXContextLogger()
        
        if verbose {
            logger.info("MLX: Starting model loading from \(url.path)")
        }
        
        do {
            let configurationURL = url.appending(component: "config.json")
            
            if verbose {
                logger.info("MLX: Loading base configuration from \(configurationURL.path)")
            }
            
            guard FileManager.default.fileExists(atPath: configurationURL.path) else {
                let errorMsg = "MLX: config.json not found at \(configurationURL.path)"
                logger.fault(errorMsg)
                throw LLMError.failedToLoad(reason: errorMsg)
            }
            
            let baseConfiguration: BaseConfiguration
            do {
                let configData = try Data(contentsOf: configurationURL)
                baseConfiguration = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
                if verbose {
                    logger.info("MLX: Successfully loaded base configuration, modelType: \(baseConfiguration.modelType)")
                }
            } catch {
                let errorMsg = "MLX: Failed to decode config.json: \(error.localizedDescription)"
                logger.fault(errorMsg)
                throw LLMError.failedToLoad(reason: errorMsg)
            }
            
            let model: any LanguageModel
            if verbose {
                logger.info("MLX: Attempting to create model of type: \(baseConfiguration.modelType)")
            }
            
            do {
                model = try VLMTypeRegistry.shared.createModel(
                    configuration: configurationURL,
                    modelType: baseConfiguration.modelType
                )
                if verbose {
                    logger.info("MLX: Successfully created VLM model")
                }
            } catch {
                if verbose {
                    logger.info("MLX: VLM model creation failed (\(error.localizedDescription)), attempting LLM fallback")
                }
                do {
                    model = try LLMTypeRegistry.shared.createModel(
                        configuration: configurationURL,
                        modelType: baseConfiguration.modelType
                    )
                    if verbose {
                        logger.info("MLX: Successfully created LLM model")
                    }
                } catch {
                    let errorMsg = "MLX: Failed to create model of type '\(baseConfiguration.modelType)': \(error.localizedDescription)"
                    logger.fault(errorMsg)
                    throw LLMError.failedToLoad(reason: errorMsg)
                }
            }

            if verbose {
                logger.info("MLX: Loading model weights from \(url.path)")
            }
            
            do {
                try loadWeights(modelDirectory: url, model: model, perLayerQuantization: baseConfiguration.perLayerQuantization)
                if verbose {
                    logger.info("MLX: Successfully loaded model weights")
                }
            } catch {
                let errorMsg = "MLX: Failed to load model weights: \(error.localizedDescription)"
                logger.fault(errorMsg)
                throw LLMError.failedToLoad(reason: errorMsg)
            }

            if verbose {
                logger.info("MLX: Loading tokenizer")
            }
            
            let tokenizer: any Tokenizer
            do {
                tokenizer = try await loadTokenizer(configuration: configuration, hub: .shared)
                if verbose {
                    logger.info("MLX: Successfully loaded tokenizer")
                }
            } catch {
                let errorMsg = "MLX: Failed to load tokenizer: \(error.localizedDescription)"
                logger.fault(errorMsg)
                throw LLMError.failedToLoad(reason: errorMsg)
            }
            
            if verbose {
                logger.info("MLX: Model loading completed successfully")
            }
            
            return (model, tokenizer)
        } catch let llmError as LLMError {
            throw llmError
        } catch {
            let errorMsg = "MLX: Unexpected error during model loading: \(error.localizedDescription)"
            logger.fault(errorMsg)
            throw LLMError.failedToLoad(reason: errorMsg)
        }
    }

    private static func makeProcessor(
        url: URL, configuration: ModelConfiguration, tokenizer: any Tokenizer, verbose: Bool = false
    ) -> (any UserInputProcessor, Bool) {
        let logger = MLXContextLogger()
        
        if verbose {
            logger.info("MLX: Creating processor for model")
        }
        
        do {
            let processorConfiguration = url.appending(
                component: "preprocessor_config.json"
            )
            
            if verbose {
                logger.info("MLX: Looking for vision processor config at \(processorConfiguration.path)")
            }
            
            let baseProcessorConfig = try JSONDecoder().decode(
                BaseProcessorConfiguration.self,
                from: Data(contentsOf: processorConfiguration)
            )
            
            if verbose {
                logger.info("MLX: Found vision processor config, type: \(baseProcessorConfig.processorClass)")
            }

            let processor = try VLMProcessorTypeRegistry.shared.createModel(
                configuration: processorConfiguration,
                processorType: baseProcessorConfig.processorClass,
                tokenizer: tokenizer
            )
            
            if verbose {
                logger.info("MLX: Successfully created vision processor - model supports vision")
            }
            
            return (processor, true)
        } catch {
            if verbose {
                logger.info("MLX: No vision processor found (\(error.localizedDescription)) - falling back to text-only processor")
            }
            return (LLMUserInputProcessor(
                tokenizer: tokenizer,
                configuration: configuration,
                messageGenerator: DefaultMessageGenerator()
            ), false)
        }
    }
}

private struct LLMUserInputProcessor: UserInputProcessor {
    let tokenizer: Tokenizer
    let configuration: ModelConfiguration
    let messageGenerator: MessageGenerator

    init(
        tokenizer: any Tokenizer, configuration: ModelConfiguration,
        messageGenerator: MessageGenerator
    ) {
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.messageGenerator = messageGenerator
    }

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        do {
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext
            )
            return LMInput(tokens: MLXArray(promptTokens))
        } catch {
            let prompt = messages
                .compactMap { $0["content"] as? String }
                .joined(separator: "\n\n")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }
    }
}

