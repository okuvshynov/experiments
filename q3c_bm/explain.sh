filename=$1
content=$(cat "$filename")

echo "Working on $filename"

curl -s http://localhost:8088/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d "$(jq -n --arg content "$content" '{
  model: "local_model",
  messages: [{
    role: "user",
    content: ("Explain what this library is doing and show some usage examples:\n" + $content)
  }]
}')" | jq -r '.timings'
