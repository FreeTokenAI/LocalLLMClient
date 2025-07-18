import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Centralized progress tracking for multi-file downloads
final class CentralizedProgressTracker: Sendable {
    let fileProgress = Locked<[String: FileProgress]>([:])
    private let progressHandler = Locked<(@Sendable (Double) async -> Void)?>(nil)
    
    struct FileProgress: Sendable {
        let expectedSize: Int64
        var downloadedSize: Int64
        var isCompleted: Bool
        
        init(expectedSize: Int64) {
            self.expectedSize = expectedSize
            self.downloadedSize = 0
            self.isCompleted = false
        }
    }
    
    func setProgressHandler(_ handler: @escaping @Sendable (Double) async -> Void) {
        progressHandler.withLock { $0 = handler }
    }
    
    func addFile(identifier: String, expectedSize: Int64) {
        fileProgress.withLock { progress in
            progress[identifier] = FileProgress(expectedSize: expectedSize)
        }
        notifyProgress()
    }
    
    func updateFileProgress(identifier: String, downloadedSize: Int64) {
        let didUpdate = fileProgress.withLock { progress in
            guard var fileProgress = progress[identifier] else { 
                return false 
            }
            fileProgress.downloadedSize = downloadedSize
            progress[identifier] = fileProgress
            return true
        }
        
        if didUpdate {
            notifyProgress()
        }
    }
    
    func markFileCompleted(identifier: String, finalSize: Int64? = nil) {
        fileProgress.withLock { progress in
            guard var fileProgress = progress[identifier] else { 
                return 
            }
            if let finalSize = finalSize {
                fileProgress.downloadedSize = finalSize
            } else {
                fileProgress.downloadedSize = fileProgress.expectedSize
            }
            fileProgress.isCompleted = true
            progress[identifier] = fileProgress
        }
        
        notifyProgress()
    }
    
    private func notifyProgress() {
        let (totalExpected, totalDownloaded, handler) = fileProgress.withLock { progress in
            let totalExpected = progress.values.reduce(0) { $0 + $1.expectedSize }
            let totalDownloaded = progress.values.reduce(0) { $0 + $1.downloadedSize }
            let handler = progressHandler.withLock { $0 }
            return (totalExpected, totalDownloaded, handler)
        }
        
        let fraction = totalExpected > 0 ? Double(totalDownloaded) / Double(totalExpected) : 0.0
        
        // Call the handler directly without Task/await to avoid main actor delays
        if let handler = handler {
            Task {
                await handler(fraction)
            }
        }
    }
}

final class Downloader {
    private(set) var downloaders: [ChildDownloader] = []
    let progress = Progress(totalUnitCount: 0)
    private let centralTracker = CentralizedProgressTracker()
    
#if os(Linux)
    private var observer: Task<Void, Never>?
#else
    private var observer: NSKeyValueObservation?
#endif

    var isDownloading: Bool {
        downloaders.contains(where: \.isDownloading)
    }

    var isDownloaded: Bool {
        downloaders.allSatisfy(\.isDownloaded)
    }
    
    var overallProgress: Double {
        let (totalExpected, totalDownloaded) = centralTracker.fileProgress.withLock { progress in
            let totalExpected = progress.values.reduce(0) { $0 + $1.expectedSize }
            let totalDownloaded = progress.values.reduce(0) { $0 + $1.downloadedSize }
            return (totalExpected, totalDownloaded)
        }
        return totalExpected > 0 ? Double(totalDownloaded) / Double(totalExpected) : 1.0
    }

    init() {}

#if os(Linux)
    deinit {
        observer?.cancel()
    }
#endif

    func add(_ downloader: ChildDownloader, expectedSize: Int64 = 1) {
        let identifier = downloader.url.absoluteString
        downloaders.append(downloader)
        
        // Register with central tracker instead of using Foundation Progress
        let tracker = centralTracker
        tracker.addFile(identifier: identifier, expectedSize: expectedSize)
        
        // Set up the downloader to report to central tracker
        downloader.setCentralTracker(tracker)
    }

    func setObserver(_ action: @Sendable @escaping (Progress) async -> Void) {
        // Instead of observing Foundation Progress, use our central tracker
        let tracker = centralTracker
        tracker.setProgressHandler { fraction in
            // Create a Progress object for compatibility
            let compatProgress = Progress(totalUnitCount: 100)
            compatProgress.completedUnitCount = Int64(fraction * 100)
            await action(compatProgress)
        }
    }

    func download() {
        guard !downloaders.isEmpty else {
            // No downloaders, nothing to do
            return
        }
        for downloader in downloaders {
            downloader.download()
        }
    }

    func waitForDownloads() async {
        // Wait until all downloads are complete
        while !downloaders.isEmpty && (isDownloading || overallProgress < 1.0) {
            try? await Task.sleep(for: .seconds(1))
        }
    }
}

extension Downloader {
    final class ChildDownloader: Sendable {
        let url: URL
        private let destinationURL: URL
        private let session: URLSession
        private let delegate: Delegate
        private let expectedSize: Int64

        var progress: Progress {
            delegate.progress
        }

        var isDownloading: Bool {
            delegate.isDownloading.withLock(\.self)
        }

        var isDownloaded: Bool {
            FileManager.default.fileExists(atPath: destinationURL.path)
        }
        
        func setCentralTracker(_ tracker: CentralizedProgressTracker) {
            delegate.setCentralTracker(tracker)
        }

        public init(url: URL, destinationURL: URL, configuration: URLSessionConfiguration = .default, expectedSize: Int64 = 0) {
            self.url = url
            self.destinationURL = destinationURL
            self.expectedSize = expectedSize
            self.delegate = Delegate(expectedSize: expectedSize, identifier: url.absoluteString)
            session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)

#if !os(Linux)
            Task {
                for task in await session.allTasks {
                    if task.taskDescription == destinationURL.absoluteString {
                        download(existingTask: task)
                    } else {
                        task.cancel()
                    }
                }
            }
#endif
        }

        public func download(existingTask: URLSessionTask? = nil) {
            guard !isDownloading else { return }
            delegate.isDownloading.withLock { $0 = true }

            try? FileManager.default.createDirectory(at: destinationURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            
            let task = existingTask ?? session.downloadTask(with: url)
            task.taskDescription = destinationURL.absoluteString
            task.priority = URLSessionTask.highPriority
            task.resume()
        }
    }
}

extension Downloader.ChildDownloader {
    final class Delegate: NSObject, URLSessionDownloadDelegate {
        let progress: Progress
        let isDownloading = Locked(false)
        private let expectedSize: Int64
        private let identifier: String
        private let centralTracker = Locked<CentralizedProgressTracker?>(nil)
        
        init(expectedSize: Int64, identifier: String) {
            self.expectedSize = expectedSize
            self.identifier = identifier
            // Initialize progress with expected size if we have it, otherwise use 1
            self.progress = Progress(totalUnitCount: max(expectedSize, 1))
            super.init()
        }
        
        func setCentralTracker(_ tracker: CentralizedProgressTracker) {
            centralTracker.withLock { $0 = tracker }
        }

        func urlSession(
            _ session: URLSession, downloadTask: URLSessionDownloadTask,
            didFinishDownloadingTo location: URL
        ) {
            // Move the downloaded file to the permanent location
            guard let destinationURL = downloadTask.destinationURL else {
                return
            }
            
            // Mark download as complete in both systems
            progress.completedUnitCount = progress.totalUnitCount
            
            // Report completion to central tracker
            let tracker = centralTracker.withLock { $0 }
            do {
                let attributes = try FileManager.default.attributesOfItem(atPath: location.path)
                let actualSize = attributes[.size] as? Int64 ?? expectedSize
                tracker?.markFileCompleted(identifier: identifier, finalSize: actualSize)
            } catch {
                tracker?.markFileCompleted(identifier: identifier)
            }
            
            try? FileManager.default.removeItem(at: destinationURL)
            do {
                try FileManager.default.createDirectory(
                    at: destinationURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try FileManager.default.moveItem(at: location, to: destinationURL)
            } catch {
                print("The URLSessionTask may be old. The app container was already invalid: \(error.localizedDescription)")
            }
        }

        func urlSession(
            _ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?
        ) {
            if error != nil {
                if let url = task.destinationURL {
                    // Attempt to remove the file if it exists
                    try? FileManager.default.removeItem(at: url)
                }
            }
            isDownloading.withLock { $0 = false }
        }

        func urlSession(
            _ session: URLSession, downloadTask: URLSessionDownloadTask,
            didWriteData bytesWritten: Int64,
            totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64
        ) {
            // Update local progress for compatibility
            progress.completedUnitCount = min(totalBytesWritten, progress.totalUnitCount)
            
            // Report to central tracker
            let tracker = centralTracker.withLock { $0 }
            tracker?.updateFileProgress(identifier: identifier, downloadedSize: totalBytesWritten)
        }
    }
}

private extension URLSessionTask {
    var destinationURL: URL? {
        guard let taskDescription else { return nil }
        return URL(string: taskDescription)
    }
}
