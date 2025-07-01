import Foundation
#if canImport(OSLog)
import OSLog
#endif

// Simple logger for MLX debugging operations
private struct MLXDebugLogger {
    #if canImport(OSLog)
    private let logger = OSLog(subsystem: "com.github.LocalLLMClient", category: "MLXDebug")
    #endif
    
    func info(_ message: String) {
        #if canImport(OSLog)
        os_log("%{public}@", log: logger, type: .info, message)
        #else
        print("[MLX DEBUG INFO] \(message)")
        #endif
    }
    
    func fault(_ message: String) {
        #if canImport(OSLog)
        os_log("%{public}@", log: logger, type: .fault, message)
        #else
        print("[MLX DEBUG FAULT] \(message)")
        #endif
    }
    
    func warning(_ message: String) {
        #if canImport(OSLog)
        os_log("%{public}@", log: logger, type: .default, message)
        #else
        print("[MLX DEBUG WARNING] \(message)")
        #endif
    }
}

/// MLX-specific logging utilities for debugging model operations
public struct MLXLogger {
    
    /// Enables console debug output for MLX operations
    /// This will print detailed information about model loading, tokenization, and generation to the console
    public static func enableConsoleDebug() {
        let logger = MLXDebugLogger()
        logger.info("MLX: Console debug logging enabled")
        print("MLX Debug: Console logging enabled. All MLX operations will be logged to the console.")
    }
    
    /// Logs detailed model directory information to help diagnose loading issues
    /// - Parameter modelURL: The URL of the model directory to inspect
    public static func logModelDirectoryInfo(at modelURL: URL) {
        let logger = MLXDebugLogger()
        
        print("=== MLX Model Directory Debug Info ===")
        print("Model Path: \(modelURL.path)")
        
        let fileManager = FileManager.default
        
        // Check if directory exists
        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: modelURL.path, isDirectory: &isDirectory)
        
        if !exists {
            print("❌ ERROR: Model directory does not exist")
            logger.fault("MLX Debug: Model directory does not exist at \(modelURL.path)")
            return
        }
        
        if !isDirectory.boolValue {
            print("❌ ERROR: Path exists but is not a directory")
            logger.fault("MLX Debug: Path exists but is not a directory: \(modelURL.path)")
            return
        }
        
        print("✅ Model directory exists")
        
        // Required files for MLX models
        let requiredFiles = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json"
        ]
        
        let optionalFiles = [
            "preprocessor_config.json",  // Vision models
            "special_tokens_map.json",
            "tokenizer.model",           // Some tokenizers
            "vocab.json",                // Some tokenizers
            "merges.txt"                 // Some tokenizers
        ]
        
        // Check required files
        print("\n--- Required Files ---")
        for file in requiredFiles {
            let filePath = modelURL.appendingPathComponent(file)
            if fileManager.fileExists(atPath: filePath.path) {
                do {
                    let attributes = try fileManager.attributesOfItem(atPath: filePath.path)
                    let size = attributes[.size] as? Int ?? 0
                    print("✅ \(file) (size: \(size) bytes)")
                } catch {
                    print("✅ \(file) (size: unknown)")
                }
            } else {
                print("❌ \(file) - MISSING")
                logger.warning("MLX Debug: Required file missing: \(file)")
            }
        }
        
        // Check optional files
        print("\n--- Optional Files ---")
        for file in optionalFiles {
            let filePath = modelURL.appendingPathComponent(file)
            if fileManager.fileExists(atPath: filePath.path) {
                do {
                    let attributes = try fileManager.attributesOfItem(atPath: filePath.path)
                    let size = attributes[.size] as? Int ?? 0
                    print("✅ \(file) (size: \(size) bytes)")
                } catch {
                    print("✅ \(file) (size: unknown)")
                }
            } else {
                print("⚪ \(file) - not present")
            }
        }
        
        // List model weight files
        print("\n--- Model Weight Files ---")
        do {
            let contents = try fileManager.contentsOfDirectory(atPath: modelURL.path)
            let weightFiles = contents.filter { $0.hasSuffix(".safetensors") || $0.hasSuffix(".gguf") }
            
            if weightFiles.isEmpty {
                print("❌ No model weight files found (.safetensors or .gguf)")
                logger.warning("MLX Debug: No model weight files found")
            } else {
                for weightFile in weightFiles.sorted() {
                    let filePath = modelURL.appendingPathComponent(weightFile)
                    do {
                        let attributes = try fileManager.attributesOfItem(atPath: filePath.path)
                        let size = attributes[.size] as? Int ?? 0
                        let sizeStr = ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)
                        print("✅ \(weightFile) (size: \(sizeStr))")
                    } catch {
                        print("✅ \(weightFile) (size: unknown)")
                    }
                }
            }
        } catch {
            print("❌ Failed to list directory contents: \(error.localizedDescription)")
            logger.fault("MLX Debug: Failed to list directory contents: \(error.localizedDescription)")
        }
        
        // Try to read and validate config.json
        print("\n--- Config Validation ---")
        let configPath = modelURL.appendingPathComponent("config.json")
        if fileManager.fileExists(atPath: configPath.path) {
            do {
                let configData = try Data(contentsOf: configPath)
                let configDict = try JSONSerialization.jsonObject(with: configData, options: []) as? [String: Any]
                
                if let modelType = configDict?["model_type"] as? String {
                    print("✅ Model type: \(modelType)")
                } else {
                    print("⚠️  Model type not found in config")
                }
                
                if let architectures = configDict?["architectures"] as? [String] {
                    print("✅ Architectures: \(architectures.joined(separator: ", "))")
                } else {
                    print("⚠️  Architectures not found in config")
                }
                
            } catch {
                print("❌ Failed to parse config.json: \(error.localizedDescription)")
                logger.fault("MLX Debug: Failed to parse config.json: \(error.localizedDescription)")
            }
        }
        
        print("=== End Debug Info ===\n")
    }
    
    /// Creates an MLXClient with verbose logging enabled for debugging
    /// - Parameters:
    ///   - url: The URL of the model directory
    ///   - parameter: Additional parameters (verbose will be set to true)
    /// - Returns: An MLXClient configured for debugging
    public static func createDebugClient(url: URL, parameter: MLXClient.Parameter = .default) async throws -> MLXClient {
        // Log model directory info first
        logModelDirectoryInfo(at: url)
        
        // Create parameter with verbose enabled
        var debugParameter = parameter
        debugParameter.options.verbose = true
        
        print("MLX Debug: Creating client with verbose logging enabled...")
        let logger = MLXDebugLogger()
        logger.info("MLX Debug: Creating client with verbose logging")
        
        return try await MLXClient(url: url, parameter: debugParameter)
    }
}