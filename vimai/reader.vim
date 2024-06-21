function! AskSonnet(question)
    let data = {}
    "let data.lines = lines
    let data.model = "claude-3-5-sonnet-20240620"
    let data.max_tokens = 1024
    let data.messages = [{"role": "user", "content": a:question}]

    " Convert the dictionary to JSON
    let json_data = json_encode(data)
    let escaped_json = substitute(json_data, "'", "'\\\\''", "g")

    let api_key = $ANTHROPIC_API_KEY

    " Construct the curl command (now pointing to localhost)
    let curl_cmd = "curl -s -X POST 'https://api.anthropic.com/v1/messages'"

    let curl_cmd .= " -H 'Content-Type: application/json'"
    let curl_cmd .= " -H 'x-api-key: " . api_key . "'"
    let curl_cmd .= " -H 'anthropic-version: 2023-06-01'"

    let curl_cmd .= " -d '" . escaped_json . "'"

    let result = system(curl_cmd)
    "echo result

    let response = json_decode(result)
    return response
endfunction

function! SendWithContext(argument)
    let user_prompt = strftime("%H:%M:%S You: ")
    " Get the start and end line numbers of the visual selection
    let [line_start, column_start] = getpos("'<")[1:2]
    let [line_end, column_end] = getpos("'>")[1:2]

    " Get the selected lines
    let lines = getline(line_start, line_end)
    echo lines

    let question = "Here's a code snippet: \n\n " . join(lines, '\n') . "\n\n" . a:argument

    let response = AskSonnet(question)

    call OpenResponseBuffer()
    call AppendToResponseBuffer(a:argument, response, user_prompt)
endfunction

function! Send(argument)
    let user_prompt = strftime("%H:%M:%S You: ")

    let response = AskSonnet(a:argument)

    call OpenResponseBuffer()
    call AppendToResponseBuffer(a:argument, response, user_prompt)
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

function! AppendToResponseBuffer(argument, response, user_prompt)
    let ai_prompt = strftime("%H:%M:%S Bot: ")
    " Move to the end of the buffer
    normal! G

    " Append a separator if the buffer is not empty
    if line('$') > 1
        call append(line('$'), repeat('-', 80))
    endif

    " Append the question
    call append(line('$'), a:user_prompt . a:argument)

    if has_key(a:response, 'content')
        let  reply = ai_prompt . a:response.content[0].text 
        call append(line('$'), split(reply, "\n"))
    else
        call append(line('$'), 'Error: ''content'' field not found in response')
    endif

    " Move the cursor to the end of the buffer
    normal! G
endfunction

" Define the command to be used in visual mode
command! -range -nargs=1 Ask :call Send(<q-args>)
command! -range -nargs=1 Askc :call SendWithContext(<q-args>)
