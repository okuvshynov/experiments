function! SendSelectedLines(argument)
    " Get the start and end line numbers of the visual selection
    let [line_start, column_start] = getpos("'<")[1:2]
    let [line_end, column_end] = getpos("'>")[1:2]

    " Get the selected lines
    let lines = getline(line_start, line_end)

    " Create a dictionary with 'lines' and 'argument'
    let data = {}
    "let data.lines = lines
    let data.model = "claude-3-5-sonnet-20240620"
    let data.max_tokens = 1024
    let data.messages = [{"role": "user", "content": a:argument}]

    " Convert the dictionary to JSON
    let json_data = json_encode(data)
    let escaped_json = substitute(json_data, "'", "'\\\\''", "g")

    " Construct the curl command (now pointing to localhost)
    "let curl_cmd = "curl -s -X POST 'http://127.0.0.1:5000/api'"
    let curl_cmd = "curl -s -X POST 'https://api.anthropic.com/v1/messages'"

    let curl_cmd .= " -H 'Content-Type: application/json'"

    let api_key = $ANTHROPIC_API_KEY
    let curl_cmd .= " -H 'x-api-key: " . api_key . "'"

    let curl_cmd .= " -H 'anthropic-version: 2023-06-01'"

    let curl_cmd .= " -d '" . escaped_json . "'"

    " Execute the curl command and capture the output
    let result = system(curl_cmd)
    echo result

    " Parse the JSON response
    let response = json_decode(result)

    call OpenResponseBuffer()
    call AppendToResponseBuffer(a:argument, response)

    " Check if the response contains the 'content' field
    if has_key(response, 'content')
        echo "Response content: " . response.content[0].text
    else
        echo "Error: 'content' field not found in response"
        echo "Full response: " . result
    endif
endfunction

function! OpenResponseBuffer()
    " Check if the buffer already exists
    let bufnum = bufnr('API_Responses')
    if bufnum == -1
        " Create a new buffer in a vertical split
        execute 'vsplit API_Responses'
        setlocal buftype=nofile
        setlocal bufhidden=hide
        setlocal noswapfile
    else
        " Check if the buffer is already displayed in a window
        let winnum = bufwinnr(bufnum)
        if winnum == -1
            " If not displayed, open it in a vertical split
            execute 'vsplit | buffer' bufnum
        else
            " If already displayed, switch to that window
            execute winnum . 'wincmd w'
        endif
    endif
endfunction

function! AppendToResponseBuffer(argument, response)
    " Move to the end of the buffer
    normal! G

    " Append a separator if the buffer is not empty
    if line('$') > 1
        call append(line('$'), repeat('-', 80))
    endif

    " Append the response
    call append(line('$'), 'Argument: ' . a:argument)
    call append(line('$'), 'Timestamp: ' . strftime("%Y-%m-%dT%H:%M:%S%z"))

    if has_key(a:response, 'content')
        call append(line('$'), 'Response content:')
        call append(line('$'), split(a:response.content[0].text, "\n"))
    else
        call append(line('$'), 'Error: ''content'' field not found in response')
    endif

    " Move the cursor to the end of the buffer
    normal! G
endfunction

" Define the command to be used in visual mode
command! -range -nargs=1 SendMe :call SendSelectedLines(<q-args>)
