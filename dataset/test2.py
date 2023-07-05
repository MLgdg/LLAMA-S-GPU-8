import tokenization as tokenization
import tokenization_glm as tokenization
import tokenization_baichuan as tokenization
import sentencepiece as spm



# sp_model = spm.SentencePieceProcessor()
# sp_model.Load('./baichuan.model')

a = "近期\n，SpaceX正在致力于进一步扩大对星际旅行的控"
# b = sp_model.encode(a, out_type=str)+['<pad>']
# print(b)
# c= sp_model.piece_to_id(b)
# print(c)

tokener = tokenization.BaiChuanTokenizer('./baichuan.model')

a = tokener.tokenize(a)
a = tokener._convert_token_to_id([])
print(a)
print(tokener.vocab_size)