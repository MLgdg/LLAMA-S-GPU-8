import tokenization as tokenization
import tokenization_glm as tokenization
import sentencepiece as spm
tokener = tokenization.SPTokenizer('./ice_text.model')
tokener2 = tokenization.TextTokenizer('./ice_text.model')

a = "徐\n某、王某民均为聋哑人。2019年1月，被告王某民以急需要资金周转为由，向原告徐某借款2万元，并承诺原告啥时候用钱啥时候还。后原告急需用钱时，经多次催要被告偿还部分借款后，尚欠原告1万元一直没有偿还，徐某于今年2月向法院提起诉讼。"
#print(tokener2.convert_tokens_to_ids(a))
#print(tokener.bos_token_id)
# print(tokener.decode([4]))
# print(tokener.num_text_tokens)
#print(tokener2.tokenize())

b=tokener.tokenize(a,add_dummy_prefix=False)  #切词 
b = tokener.convert_tokens_to_ids(b)  #词转化id
#b = tokener.decode([6]) #id 转为词把很多特殊token都省略了
pad = tokener.pad_token_id
bos = tokener.bos_token_id#文本的开始
end = tokener.end_token_id
eos = tokener.eos_token_id # End of Program
print(end)
print(tokener[end])