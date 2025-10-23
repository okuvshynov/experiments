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
    var toolbar: NSToolbar!

    @objc func addTextAnnotation(_ sender: Any) {
        guard let currentPage = pdfView.currentPage else { return }

        // Create annotation at center of visible area
        let bounds = NSRect(x: 100, y: 100, width: 200, height: 100)
        let annotation = PDFAnnotation(bounds: bounds, forType: .freeText, withProperties: nil)
        annotation.contents = "Double-click to edit"
        annotation.color = NSColor.yellow.withAlphaComponent(0.3)

        currentPage.addAnnotation(annotation)
        logAnnotation(annotation)
    }

    @objc func addNoteAnnotation(_ sender: Any) {
        guard let currentPage = pdfView.currentPage else { return }

        let bounds = NSRect(x: 150, y: 150, width: 20, height: 20)
        let annotation = PDFAnnotation(bounds: bounds, forType: .text, withProperties: nil)
        annotation.contents = "Note annotation"
        annotation.color = NSColor.systemYellow

        currentPage.addAnnotation(annotation)
        logAnnotation(annotation)
    }

    @objc func addHighlightAnnotation(_ sender: Any) {
        // Create highlight from current selection
        guard let selection = pdfView.currentSelection,
              let page = selection.pages.first else {
            print("INFO: Select text first to create a highlight annotation")
            fflush(stdout)
            return
        }

        let selections = selection.selectionsByLine()
        guard let firstSelection = selections.first else { return }

        let bounds = firstSelection.bounds(for: page)
        let annotation = PDFAnnotation(bounds: bounds, forType: .highlight, withProperties: nil)
        annotation.color = NSColor.yellow.withAlphaComponent(0.5)

        page.addAnnotation(annotation)
        logAnnotation(annotation)
    }

    @objc func addCircleAnnotation(_ sender: Any) {
        guard let currentPage = pdfView.currentPage else { return }

        let bounds = NSRect(x: 200, y: 200, width: 100, height: 100)
        let annotation = PDFAnnotation(bounds: bounds, forType: .circle, withProperties: nil)
        annotation.color = NSColor.red.withAlphaComponent(0.3)

        currentPage.addAnnotation(annotation)
        logAnnotation(annotation)
    }

    func logAnnotation(_ annotation: PDFAnnotation) {
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
        print("Annotation Controls:")
        print("  Cmd+T - Add Text Annotation")
        print("  Cmd+N - Add Note Annotation")
        print("  Cmd+H - Add Highlight (select text first)")
        print("  Cmd+R - Add Circle Annotation")
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

        window.makeKeyAndOrderFront(nil)

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

        // Application menu
        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)
        let appMenu = NSMenu()
        appMenuItem.submenu = appMenu
        appMenu.addItem(withTitle: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        // Annotations menu
        let annotationsMenuItem = NSMenuItem()
        annotationsMenuItem.title = "Annotations"
        mainMenu.addItem(annotationsMenuItem)

        let annotationsMenu = NSMenu(title: "Annotations")
        annotationsMenuItem.submenu = annotationsMenu

        annotationsMenu.addItem(withTitle: "Add Text Annotation", action: #selector(addTextAnnotation(_:)), keyEquivalent: "t")
        annotationsMenu.addItem(withTitle: "Add Note", action: #selector(addNoteAnnotation(_:)), keyEquivalent: "n")
        annotationsMenu.addItem(withTitle: "Add Highlight", action: #selector(addHighlightAnnotation(_:)), keyEquivalent: "h")
        annotationsMenu.addItem(withTitle: "Add Circle", action: #selector(addCircleAnnotation(_:)), keyEquivalent: "r")

        NSApp.mainMenu = mainMenu
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

// Main entry point
let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
