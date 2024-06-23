let g:vqq_api_key    = get(g:, 'vqq_api_key'   , $ANTHROPIC_API_KEY)
let g:vqq_max_tokens = get(g:, 'vqq_max_tokens', 1024)
let g:vqq_model_name = get(g:, 'vqq_model_name', "claude-3-5-sonnet-20240620")
let g:vqq_local_addr = get(g:, 'vqq_local_host', "http://studio.local:8080/chat/completions")
let g:vqq_backend    = get(g:, 'vqq_backend'   , "anthropic")

let s:job_id = 0
let s:curr   = []

" TODO - assumes one query at a time for now
function! s:on_stdout(channel, msg)
    call add(s:curr, a:msg)
endfunction

function! s:on_exit(job_id, exit_status)
    let response = json_decode(join(s:curr, '\n'))
    call s:open_chat()
    let ai_prompt = strftime("%H:%M:%S Bot: ")
    let reply     = ai_prompt . s:parse_impl(response)
    normal! G
    setlocal modifiable
    call append(line('$'), split(reply, "\n"))
    setlocal nomodifiable
    normal! G
endfunction

function! s:ask_local(question)
    let req = {}
    let req.n_predict = g:vqq_max_tokens
    let req.messages   = [{"role": "user", "content": a:question}]

    let json_req = json_encode(req)
    let json_req = substitute(json_req, "'", "'\\\\''", "g")

    let curl_cmd  = "curl -s -X POST '" . g:vqq_local_addr . "'"
    let curl_cmd .= " -H 'Content-Type: application/json'"
    let curl_cmd .= " -d '" . json_req . "'"

    let s:curr = []
    let s:job_id = job_start(['/bin/sh', '-c', curl_cmd], {'out_cb': 's:on_stdout', 'exit_cb': 's:on_exit'})

endfunction

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

    let s:job_id = job_start(['/bin/sh', '-c', curl_cmd], {'out_cb': 's:on_stdout', 'exit_cb': 's:on_exit'})

endfunction

function! s:parse_anthropic(response)
    " TODO: error handling.
    return a:response.content[0].text
endfunction

function! s:parse_local(response)
    return a:response.choices[0].message.content
endfunction

let s:backend_impl = {
  \ 'anthropic' : {
  \   'ask'  : function('s:ask_anthropic'),
  \   'parse': function('s:parse_anthropic'),
  \ },
  \ 'local' : {
  \   'ask'  : function('s:ask_local'),
  \   'parse': function('s:parse_local'),
  \ },
\ }

function! s:ask_impl(question)
    return s:backend_impl[g:vqq_backend]['ask'](a:question)
endfunction

function! s:parse_impl(reply)
    return s:backend_impl[g:vqq_backend]['parse'](a:reply)
endfunction

function! s:ask(argument)
    call s:add_question(a:argument)
    call s:ask_impl(a:argument)
endfunction

function! s:ask_with_context(argument)
    " Get the selected lines
    let [line_a, column_a] = getpos("'<")[1:2]
    let [line_b, column_b] = getpos("'>")[1:2]
    let lines = getline(line_a, line_b)

    " Basic prompt format
    let question = "Here's a code snippet: \n\n " . join(lines, '\n') . "\n\n" . a:argument
    call s:add_question(a:argument)
    call s:ask_impl(question)

    " Not passing the context to the chat window, only the question
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
        setlocal nomodifiable
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

function! s:add_question(question)
    call s:open_chat()

    let prompt = strftime("%H:%M:%S You: ")
    normal! G

    setlocal modifiable
    if line('$') > 1
        call append(line('$'), repeat('-', 80))
    endif

    call append(line('$'), prompt . a:question)
    setlocal nomodifiable

    normal! G
endfunction

command! -nargs=1 QQ :call s:ask(<q-args>)
command! -range -nargs=1 QQQ :call s:ask_with_context(<q-args>)
