import tokenization as tokenization
import tokenization_glm as tokenization
import sentencepiece as spm
tokener = tokenization.SPTokenizer('./ice_text.model')
tokener2 = tokenization.TextTokenizer('./ice_text.model')

a = "中华人民共和国。，。，。，。\nfuck <0xFF>"
#print(tokener2.convert_tokens_to_ids(a))
#print(tokener.bos_token_id)
# print(tokener.decode([4]))
# print(tokener.num_text_tokens)
#print(tokener2.tokenize())

b=tokener.tokenize(a,add_dummy_prefix=False)  #切词 
b = tokener.convert_tokens_to_ids(b)  #词转化id
#b = tokener.decode([6]) #id 转为词把很多特殊token都省略了
pad = tokener.pad_token_id
bos = tokener.bos_token_id
end = tokener.end_token_id
print(b)
print(tokener[130343])