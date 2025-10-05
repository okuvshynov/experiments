package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
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

var (
	config       Config
	proxyPort    string
	backendURL   string
	configFile   string
	placeholder  string
)

func init() {
	flag.StringVar(&proxyPort, "port", "8081", "Proxy server port")
	flag.StringVar(&backendURL, "backend", "http://localhost:8080", "Backend LLM server URL")
	flag.StringVar(&configFile, "config", "config.json", "Config file with prefix mappings")
	flag.StringVar(&placeholder, "placeholder", "__the__user__message__", "Placeholder in template files")
}

func main() {
	flag.Parse()

	// Load config
	if err := loadConfig(configFile); err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	log.Printf("Loaded config with %d prefix mappings\n", len(config.Prefixes))

	// Parse backend URL
	target, err := url.Parse(backendURL)
	if err != nil {
		log.Fatalf("Invalid backend URL: %v", err)
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

func handleChatCompletions(w http.ResponseWriter, r *http.Request, target *url.URL) {
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

				// Strip prefix from message
				actualMessage := strings.TrimPrefix(content, prefix)
				actualMessage = strings.TrimSpace(actualMessage)

				// Replace placeholder with actual message
				injectedContent := strings.ReplaceAll(string(template), placeholder, actualMessage)

				// Update message content
				messages[i].Content = injectedContent
				modified = true
				log.Printf("Injected template %s for prefix %s\n", templateFile, prefix)
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
