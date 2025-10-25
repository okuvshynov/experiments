import Cocoa
import PDFKit
import Foundation

// Observer class to handle PDF view selection events
class PDFSelectionObserver: NSObject {
    @objc func pdfViewSelectionChanged(_ notification: Notification) {
        guard let pdfView = notification.object as? PDFView,
              let currentSelection = pdfView.currentSelection,
              let selectedText = currentSelection.string else {
            return
        }

        if !selectedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            print("TEXT_SELECTION: \(selectedText)")
            fflush(stdout)
        }
    }
}

// Observer for annotation changes
class AnnotationObserver: NSObject {
    @objc func annotationChanged(_ notification: Notification) {
        guard let annotation = notification.userInfo?["PDFAnnotationKey"] as? PDFAnnotation else {
            return
        }

        var annotationInfo = "ANNOTATION_CREATED: Type=\(annotation.type ?? "unknown")"

        if let contents = annotation.contents, !contents.isEmpty {
            annotationInfo += ", Contents=\"\(contents)\""
        }

        if let page = annotation.page {
            annotationInfo += ", Page=\(page.label ?? "unknown")"
        }

        annotationInfo += ", Bounds=\(annotation.bounds)"

        print(annotationInfo)
        fflush(stdout)
    }
}

// Application delegate
class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var pdfView: PDFView!
    var selectionObserver: PDFSelectionObserver!
    var annotationObserver: AnnotationObserver!
    @objc func addComment(_ sender: Any) {
        // Must have a text selection
        guard let selection = pdfView.currentSelection,
              let page = selection.pages.first,
              let selectedText = selection.string else {
            print("INFO: Select text first to add a comment")
            fflush(stdout)
            return
        }

        // Show input dialog for comment
        let alert = NSAlert()
        alert.messageText = "Add Comment"
        alert.informativeText = "Enter your comment for the selected text:"
        alert.addButton(withTitle: "Add")
        alert.addButton(withTitle: "Cancel")

        let inputTextField = NSTextField(frame: NSRect(x: 0, y: 0, width: 300, height: 24))
        inputTextField.placeholderString = "Your comment here..."
        alert.accessoryView = inputTextField

        alert.window.initialFirstResponder = inputTextField

        let response = alert.runModal()
        if response == .alertFirstButtonReturn {
            let commentText = inputTextField.stringValue
            guard !commentText.isEmpty else { return }

            // Create highlight annotation with the comment
            let selections = selection.selectionsByLine()
            guard let firstSelection = selections.first else { return }

            let bounds = firstSelection.bounds(for: page)
            let annotation = PDFAnnotation(bounds: bounds, forType: .highlight, withProperties: nil)
            annotation.color = NSColor.yellow.withAlphaComponent(0.5)
            annotation.contents = commentText

            page.addAnnotation(annotation)
            logAnnotation(annotation, selectedText: selectedText)
        }
    }

    func logAnnotation(_ annotation: PDFAnnotation, selectedText: String) {
        var annotationInfo = "ANNOTATION_CREATED:"

        if let page = annotation.page {
            annotationInfo += " Page=\(page.label ?? "unknown"),"
        }

        annotationInfo += " SelectedText=\"\(selectedText)\","

        if let contents = annotation.contents, !contents.isEmpty {
            annotationInfo += " Comment=\"\(contents)\""
        }

        print(annotationInfo)
        fflush(stdout)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Get PDF path from command line arguments
        let args = CommandLine.arguments
        guard args.count > 1 else {
            print("ERROR: No PDF file specified")
            print("Usage: pdfq <path/to/file.pdf>")
            NSApp.terminate(nil)
            return
        }

        let pdfPath = args[1]
        let fileURL = URL(fileURLWithPath: pdfPath)

        // Check if file exists
        guard FileManager.default.fileExists(atPath: pdfPath) else {
            print("ERROR: File not found: \(pdfPath)")
            NSApp.terminate(nil)
            return
        }

        // Load PDF document
        guard let pdfDocument = PDFDocument(url: fileURL) else {
            print("ERROR: Failed to load PDF document: \(pdfPath)")
            NSApp.terminate(nil)
            return
        }

        print("Loaded PDF: \(pdfPath)")
        print("Pages: \(pdfDocument.pageCount)")
        print("How to use:")
        print("  - Select text to see it logged")
        print("  - Press Cmd+K to add a comment to selected text")
        print("---")
        fflush(stdout)

        // Create window
        let windowRect = NSRect(x: 100, y: 100, width: 800, height: 1000)
        window = NSWindow(contentRect: windowRect,
                         styleMask: [.titled, .closable, .miniaturizable, .resizable],
                         backing: .buffered,
                         defer: false)
        window.title = "pdfq - \(fileURL.lastPathComponent)"

        // Create menu bar
        setupMenuBar()

        // Show and activate window
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        // Create PDF view
        pdfView = PDFView(frame: windowRect)
        pdfView.document = pdfDocument
        pdfView.autoScales = true
        pdfView.displayMode = .singlePageContinuous
        pdfView.displayDirection = .vertical

        // Set up scrolling
        let scrollView = NSScrollView(frame: windowRect)
        scrollView.documentView = pdfView
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = true
        scrollView.autoresizingMask = [.width, .height]

        window.contentView = scrollView

        // Set up observer for text selection
        selectionObserver = PDFSelectionObserver()

        // Register for selection change notifications
        NotificationCenter.default.addObserver(
            selectionObserver!,
            selector: #selector(PDFSelectionObserver.pdfViewSelectionChanged(_:)),
            name: .PDFViewSelectionChanged,
            object: pdfView
        )

        // Set up annotation observer
        annotationObserver = AnnotationObserver()
        NotificationCenter.default.addObserver(
            annotationObserver!,
            selector: #selector(AnnotationObserver.annotationChanged(_:)),
            name: .PDFDocumentDidEndPageWrite,
            object: pdfDocument
        )

        // Also observe annotation hit notification which fires when annotation is added
        NotificationCenter.default.addObserver(
            annotationObserver!,
            selector: #selector(AnnotationObserver.annotationChanged(_:)),
            name: NSNotification.Name("PDFAnnotationChanged"),
            object: nil
        )

        // Observe page changes for new annotations
        NotificationCenter.default.addObserver(
            annotationObserver!,
            selector: #selector(AnnotationObserver.annotationChanged(_:)),
            name: .PDFDocumentDidEndPageFind,
            object: pdfDocument
        )
    }

    func setupMenuBar() {
        let mainMenu = NSMenu()

        // Application menu (pdfq)
        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)
        let appMenu = NSMenu(title: "pdfq")
        appMenuItem.submenu = appMenu

        appMenu.addItem(withTitle: "About pdfq", action: nil, keyEquivalent: "")
        appMenu.addItem(NSMenuItem.separator())
        appMenu.addItem(withTitle: "Quit pdfq", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        // File menu
        let fileMenuItem = NSMenuItem()
        mainMenu.addItem(fileMenuItem)
        let fileMenu = NSMenu(title: "File")
        fileMenuItem.submenu = fileMenu
        fileMenu.addItem(withTitle: "Close Window", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")

        // Comment menu
        let commentMenuItem = NSMenuItem()
        mainMenu.addItem(commentMenuItem)
        let commentMenu = NSMenu(title: "Comment")
        commentMenuItem.submenu = commentMenu

        let addCommentItem = commentMenu.addItem(withTitle: "Add Comment to Selection", action: #selector(addComment(_:)), keyEquivalent: "k")
        addCommentItem.target = self

        NSApp.mainMenu = mainMenu
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

// Main entry point
let app = NSApplication.shared

// Set activation policy to regular app (shows in dock, has menu bar)
app.setActivationPolicy(.regular)

let delegate = AppDelegate()
app.delegate = delegate

// Activate the app and bring to front
app.activate(ignoringOtherApps: true)

app.run()
