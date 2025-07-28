filename=$1
content=$(cat "$filename")

curl -s http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d "$(jq -n --arg content "$content" '{
  model: "UD-IQ1_M",
  cache_prompt: false,
  messages: [{
    role: "user",
    content: ("Find potential compile error in this file. Show both the line with error and how it should be fixed\n" + $content)
  }]
}')" | jq -r '.timings'
