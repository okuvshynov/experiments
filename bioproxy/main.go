package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"
)

// Config represents the prefix to template file mappings
type Config struct {
	Prefixes map[string]string `json:"prefixes"`
}

// ChatMessage represents a message in the OpenAI format
type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // Can be string or array
}

// ChatRequest represents the incoming chat completion request
type ChatRequest struct {
	Messages []ChatMessage          `json:"messages"`
	Other    map[string]interface{} `json:"-"`
}

// WarmupState tracks warmup status and file changes
type WarmupState struct {
	mu               sync.Mutex
	lastPrefix       string            // Last used template prefix
	fileHashes       map[string]string // filepath -> content hash
	activeRequest    bool              // True if main request is active
	warmupInProgress bool              // True if warmup is running
	warmupPending    bool              // True if another warmup needed after current
}

var (
	config         Config
	warmupState    *WarmupState
	proxyPort      string
	backendURL     string
	configFile     string
	placeholder    string
	warmupInterval time.Duration
)

func init() {
	flag.StringVar(&proxyPort, "port", "8081", "Proxy server port")
	flag.StringVar(&backendURL, "backend", "http://localhost:8080", "Backend LLM server URL")
	flag.StringVar(&configFile, "config", "config.json", "Config file with prefix mappings")
	flag.StringVar(&placeholder, "placeholder", "__the__user__message__", "Placeholder in template files")
	flag.DurationVar(&warmupInterval, "warmup-interval", 30*time.Second, "Interval between warmup checks (0 to disable)")
}

func main() {
	flag.Parse()

	// Load config
	if err := loadConfig(configFile); err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	log.Printf("Loaded config with %d prefix mappings\n", len(config.Prefixes))

	// Initialize warmup state
	warmupState = &WarmupState{
		fileHashes: make(map[string]string),
	}

	// Parse backend URL
	target, err := url.Parse(backendURL)
	if err != nil {
		log.Fatalf("Invalid backend URL: %v", err)
	}

	// Start warmup checker if enabled
	if warmupInterval > 0 {
		log.Printf("Warmup enabled with interval: %v\n", warmupInterval)
		go warmupChecker(target)
	} else {
		log.Printf("Warmup disabled\n")
	}

	// Create reverse proxy for non-chat endpoints
	proxy := httputil.NewSingleHostReverseProxy(target)

	// Handle /v1/chat/completions with message injection
	http.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		handleChatCompletions(w, r, target)
	})

	// Handle /chat/completions as well (some servers use this)
	http.HandleFunc("/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		handleChatCompletions(w, r, target)
	})

	// Proxy all other requests unchanged
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		proxy.ServeHTTP(w, r)
	})

	log.Printf("Proxy server listening on :%s, forwarding to %s\n", proxyPort, backendURL)
	log.Fatal(http.ListenAndServe(":"+proxyPort, nil))
}

func loadConfig(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, &config)
}

// calculateFileHash computes SHA256 hash of file content
func calculateFileHash(filepath string) (string, error) {
	content, err := os.ReadFile(filepath)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(content)
	return fmt.Sprintf("%x", hash), nil
}

// extractIncludedFiles finds all <{filepath}> references in template
func extractIncludedFiles(template string) []string {
	re := regexp.MustCompile(`<\{([^}]+)\}>`)
	matches := re.FindAllStringSubmatch(template, -1)

	files := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) >= 2 {
			files = append(files, strings.TrimSpace(match[1]))
		}
	}
	return files
}

// getTemplateFiles returns all files involved in a template (template + includes)
func getTemplateFiles(templateFile string) ([]string, error) {
	files := []string{templateFile}

	// Read template to find includes
	content, err := os.ReadFile(templateFile)
	if err != nil {
		return nil, err
	}

	// Add all included files
	included := extractIncludedFiles(string(content))
	files = append(files, included...)

	return files, nil
}

// filesChanged checks if any files have changed since last check
func filesChanged(files []string, oldHashes map[string]string) (bool, map[string]string) {
	newHashes := make(map[string]string)
	changed := false

	for _, file := range files {
		hash, err := calculateFileHash(file)
		if err != nil {
			// File might not exist yet, treat as changed
			log.Printf("Warning: Could not hash file %s: %v\n", file, err)
			changed = true
			continue
		}

		newHashes[file] = hash

		// Check if hash is different
		if oldHash, exists := oldHashes[file]; !exists || oldHash != hash {
			changed = true
		}
	}

	// Also check if any old files were removed
	for oldFile := range oldHashes {
		if _, exists := newHashes[oldFile]; !exists {
			changed = true
		}
	}

	return changed, newHashes
}

// sendWarmupRequest sends a warmup call with n_predict=0
func sendWarmupRequest(prefix, templateFile string, target *url.URL) error {
	// Read and process template
	template, err := os.ReadFile(templateFile)
	if err != nil {
		return fmt.Errorf("failed to read template: %w", err)
	}

	processedTemplate, err := processTemplateIncludes(string(template))
	if err != nil {
		log.Printf("Warning: Failed to process includes in warmup template: %v\n", err)
		processedTemplate = string(template)
	}

	// Replace placeholder with empty string
	content := strings.ReplaceAll(processedTemplate, placeholder, "")

	// Build warmup request
	requestBody := map[string]interface{}{
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": content,
			},
		},
		"n_predict": 0, // Just process prompt, don't generate
	}

	bodyJSON, err := json.Marshal(requestBody)
	if err != nil {
		return fmt.Errorf("failed to marshal warmup request: %w", err)
	}

	// Send to backend
	resp, err := http.Post(
		target.String()+"/v1/chat/completions",
		"application/json",
		bytes.NewReader(bodyJSON),
	)
	if err != nil {
		return fmt.Errorf("failed to send warmup request: %w", err)
	}
	defer resp.Body.Close()

	// Discard response body
	io.Copy(io.Discard, resp.Body)

	log.Printf("Warmup request sent for prefix %s (status: %d)\n", prefix, resp.StatusCode)
	return nil
}

// warmupChecker periodically checks for file changes and sends warmup requests
func warmupChecker(target *url.URL) {
	ticker := time.NewTicker(warmupInterval)
	defer ticker.Stop()

	for range ticker.C {
		performWarmupIfNeeded(target)
	}
}

// performWarmupIfNeeded checks for changes and sends warmup, handling pending warmups
func performWarmupIfNeeded(target *url.URL) {
	for {
		warmupState.mu.Lock()

		// Skip if no template has been used yet
		if warmupState.lastPrefix == "" {
			warmupState.mu.Unlock()
			return
		}

		// Skip if main request is active
		if warmupState.activeRequest {
			warmupState.mu.Unlock()
			return
		}

		// Get template file for last used prefix
		templateFile, exists := config.Prefixes[warmupState.lastPrefix]
		if !exists {
			warmupState.mu.Unlock()
			return
		}

		// Get all files (template + includes)
		files, err := getTemplateFiles(templateFile)
		if err != nil {
			log.Printf("Warning: Could not get template files for warmup: %v\n", err)
			warmupState.mu.Unlock()
			return
		}

		// Always check if files changed (even if warmup in progress)
		changed, newHashes := filesChanged(files, warmupState.fileHashes)

		if warmupState.warmupInProgress {
			// Warmup already running
			if changed {
				// Mark pending and update hashes for next warmup
				warmupState.warmupPending = true
				warmupState.fileHashes = newHashes
				log.Printf("Warmup in progress, files changed - marking pending for prefix %s\n", warmupState.lastPrefix)
			}
			warmupState.mu.Unlock()
			return
		}

		// No warmup in progress - check if we should start one
		if changed || warmupState.warmupPending {
			// Start warmup
			warmupState.warmupInProgress = true
			warmupState.warmupPending = false
			warmupState.fileHashes = newHashes
			prefix := warmupState.lastPrefix
			warmupState.mu.Unlock()

			// Send warmup (outside lock to avoid blocking)
			if changed {
				log.Printf("Files changed, sending warmup for prefix %s\n", prefix)
			} else {
				log.Printf("Processing pending warmup for prefix %s\n", prefix)
			}

			err := sendWarmupRequest(prefix, templateFile, target)
			if err != nil {
				log.Printf("Warning: Warmup failed: %v\n", err)
			}

			// Clear in-progress flag and check if another warmup is pending
			warmupState.mu.Lock()
			warmupState.warmupInProgress = false
			stillPending := warmupState.warmupPending
			warmupState.mu.Unlock()

			// If another warmup is pending, loop immediately
			if stillPending {
				log.Printf("Another warmup pending, processing immediately\n")
				continue
			}
		} else {
			warmupState.mu.Unlock()
		}

		// No more warmups needed
		return
	}
}

// processTemplateIncludes replaces <{filepath}> placeholders with file contents
func processTemplateIncludes(template string) (string, error) {
	// Match <{filepath}> pattern
	re := regexp.MustCompile(`<\{([^}]+)\}>`)

	result := template
	matches := re.FindAllStringSubmatch(template, -1)

	for _, match := range matches {
		if len(match) < 2 {
			continue
		}

		placeholder := match[0] // Full match: <{filepath}>
		filepath := strings.TrimSpace(match[1]) // Just the filepath

		// Read the file
		content, err := os.ReadFile(filepath)
		if err != nil {
			log.Printf("Warning: Failed to read included file %s: %v\n", filepath, err)
			// Replace with error message instead of failing
			result = strings.ReplaceAll(result, placeholder, fmt.Sprintf("[Error reading %s: %v]", filepath, err))
			continue
		}

		// Replace placeholder with file content
		result = strings.ReplaceAll(result, placeholder, string(content))
		log.Printf("Included file %s into template\n", filepath)
	}

	return result, nil
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request, target *url.URL) {
	// Mark request as active
	warmupState.mu.Lock()
	warmupState.activeRequest = true
	warmupState.mu.Unlock()

	// Ensure we clear active flag when done
	defer func() {
		warmupState.mu.Lock()
		warmupState.activeRequest = false
		warmupState.mu.Unlock()
	}()

	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// Parse JSON into generic map first to preserve all fields
	var rawRequest map[string]interface{}
	if err := json.Unmarshal(body, &rawRequest); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Extract messages
	messagesRaw, ok := rawRequest["messages"]
	if !ok {
		http.Error(w, "Missing messages field", http.StatusBadRequest)
		return
	}

	messagesJSON, _ := json.Marshal(messagesRaw)
	var messages []ChatMessage
	if err := json.Unmarshal(messagesJSON, &messages); err != nil {
		http.Error(w, "Invalid messages format", http.StatusBadRequest)
		return
	}

	// Check if we need to inject context
	modified := false
	for i, msg := range messages {
		if msg.Role != "user" {
			continue
		}

		// Get message content as string
		content := getContentAsString(msg.Content)
		if content == "" {
			continue
		}

		// Check for prefix match
		for prefix, templateFile := range config.Prefixes {
			if strings.HasPrefix(content, prefix) {
				// Read template file (always fresh)
				template, err := os.ReadFile(templateFile)
				if err != nil {
					log.Printf("Warning: Failed to read template %s: %v\n", templateFile, err)
					continue
				}

				// Process any <{filepath}> includes in the template
				processedTemplate, err := processTemplateIncludes(string(template))
				if err != nil {
					log.Printf("Warning: Failed to process includes in template %s: %v\n", templateFile, err)
					processedTemplate = string(template) // Fall back to original
				}

				// Strip prefix from message
				actualMessage := strings.TrimPrefix(content, prefix)
				actualMessage = strings.TrimSpace(actualMessage)

				// Replace placeholder with actual message
				injectedContent := strings.ReplaceAll(processedTemplate, placeholder, actualMessage)

				// Update message content
				messages[i].Content = injectedContent
				modified = true
				log.Printf("Injected template %s for prefix %s\n", templateFile, prefix)

				// Update warmup state with this template
				go func(p, tf string) {
					files, err := getTemplateFiles(tf)
					if err != nil {
						log.Printf("Warning: Could not track template files: %v\n", err)
						return
					}

					_, hashes := filesChanged(files, nil)

					warmupState.mu.Lock()
					warmupState.lastPrefix = p
					warmupState.fileHashes = hashes
					warmupState.mu.Unlock()
				}(prefix, templateFile)

				break
			}
		}

		// Only process first user message
		break
	}

	// If modified, update the messages in the raw request
	if modified {
		rawRequest["messages"] = messages
		body, _ = json.Marshal(rawRequest)
	}

	// Create new request to backend
	backendReq, err := http.NewRequest(r.Method, target.String()+r.URL.Path, bytes.NewReader(body))
	if err != nil {
		http.Error(w, "Failed to create backend request", http.StatusInternalServerError)
		return
	}

	// Copy headers
	for key, values := range r.Header {
		for _, value := range values {
			backendReq.Header.Add(key, value)
		}
	}

	// Update Content-Length if body was modified
	if modified {
		backendReq.ContentLength = int64(len(body))
		backendReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(body)))
	}

	// Forward request
	client := &http.Client{}
	resp, err := client.Do(backendReq)
	if err != nil {
		http.Error(w, "Failed to reach backend", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	// Copy response status and body
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func getContentAsString(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		// Handle multimodal content (array of objects)
		// Look for text type
		for _, item := range v {
			if obj, ok := item.(map[string]interface{}); ok {
				if typeVal, hasType := obj["type"]; hasType && typeVal == "text" {
					if text, hasText := obj["text"].(string); hasText {
						return text
					}
				}
			}
		}
	}
	return ""
}
