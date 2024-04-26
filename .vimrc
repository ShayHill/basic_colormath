vim9script

# run the main python file with
nnoremap <leader>m :update<CR>:ScratchTermReplaceU python src/basic_colormath2/main.py<CR>
inoremap <leader>m <esc>:update<CR>:ScratchTermReplaceU python src/basic_colormath2/main.py<CR>

# resize the code window after running a python file
nnoremap <leader>w :resize 65<CR>
