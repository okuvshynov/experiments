let g:vqq_api_key    = get(g:, 'vqq_api_key'   , $ANTHROPIC_API_KEY)
let g:vqq_max_tokens = get(g:, 'vqq_max_tokens', 1024)
let g:vqq_model_name = get(g:, 'vqq_model_name', "claude-3-5-sonnet-20240620")

function! s:ask_anthropic(question)
    let req = {}
    let req.model      = g:vqq_model_name
    let req.max_tokens = g:vqq_max_tokens
    let req.messages   = [{"role": "user", "content": a:question}]

    let json_req = json_encode(req)
    let json_req = substitute(json_req, "'", "'\\\\''", "g")

    let curl_cmd  = "curl -s -X POST 'https://api.anthropic.com/v1/messages'"
    let curl_cmd .= " -H 'Content-Type: application/json'"
    let curl_cmd .= " -H 'x-api-key: " . g:vqq_api_key . "'"
    let curl_cmd .= " -H 'anthropic-version: 2023-06-01'"
    let curl_cmd .= " -d '" . json_req . "'"

    let json_res = system(curl_cmd)

    return json_decode(json_res)
endfunction

function! s:ask(argument)
    let user_prompt = strftime("%H:%M:%S You: ")

    let response = s:ask_anthropic(a:argument)

    call s:update_chat(a:argument, response, user_prompt)
endfunction

function! s:ask_with_context(argument)
    let user_prompt = strftime("%H:%M:%S You: ")
    " Get the selected lines
    let [line_a, column_a] = getpos("'<")[1:2]
    let [line_b, column_b] = getpos("'>")[1:2]
    let lines = getline(line_a, line_b)

    " Basic prompt format
    let question = "Here's a code snippet: \n\n " . join(lines, '\n') . "\n\n" . a:argument
    let response = s:ask_anthropic(question)

    " Not passing the context to the chat window, only the question
    call s:update_chat(a:argument, response, user_prompt)
endfunction

function! s:open_chat()
    " Check if the buffer already exists
    let bufnum = bufnr('VQQ_Chat')
    if bufnum == -1
        " Create a new buffer in a vertical split
        execute 'vsplit VQQ_Chat'
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

function! s:update_chat(argument, response, user_prompt)
    call s:open_chat()

    let ai_prompt = strftime("%H:%M:%S Bot: ")
    normal! G

    if line('$') > 1
        call append(line('$'), repeat('-', 80))
    endif

    " Append the question
    call append(line('$'), a:user_prompt . a:argument)

    " Append the reply
    " TODO: error handling. this format is anthropic-specific
    if has_key(a:response, 'content')
        let  reply = ai_prompt . a:response.content[0].text 
        call append(line('$'), split(reply, "\n"))
    else
        call append(line('$'), 'Error: ''content'' field not found in response')
    endif

    normal! G
endfunction

command! -nargs=1 QQ :call s:ask(<q-args>)
command! -range -nargs=1 QQQ :call s:ask_with_context(<q-args>)
