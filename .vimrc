vim9script

# run the main python file
nnoremap <leader>m :update<CR>:ScratchTermReplaceU .venv/Scripts/python.exe src/basic_colormath/main.py<CR>
nnoremap <leader>l :call g:RunPrecommit()<CR>
nnoremap <leader>L :call g:RunPrecommitAll()<CR>

set grepprg=rg\ --vimgrep\ --no-heading\ --glob\ !binaries\ --glob\ !resources\ --glob\ !uv.lock
