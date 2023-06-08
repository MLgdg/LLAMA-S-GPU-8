import sentencepiece as spm

class TextTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.num_tokens = self.sp.vocab_size()

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, ids):
        return self.sp.DecodeIds(ids)

    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)

    def convert_tokens_to_string(self, tokens):
        return self.sp.DecodePieces(tokens)

    def convert_tokens_to_ids(self, tokens):
        return [self.sp.PieceToId(token) for token in tokens]

    def convert_token_to_id(self, token):
        return self.sp.PieceToId(token)

    def convert_id_to_token(self, idx):
        return self.sp.IdToPiece(idx)

    def __len__(self):
        return self.num_tokens

class SPTokenizer:
    def __init__(
            self,
            vocab_file,
            num_image_tokens=0,
            max_blank_length=80,
            byte_fallback=True,
    ):
        assert vocab_file is not None
        self.vocab_file = vocab_file
        self.num_image_tokens = num_image_tokens
        self.special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "<unused_0>", "<sop>", "<eop>", "<ENC>", "<dBLOCK>"]
        self.max_blank_length = max_blank_length
        self.byte_fallback = byte_fallback
        self.text_tokenizer = TextTokenizer(vocab_file)

    @property
    def bos_token_id(self):
        return self["<sop>"]

    @property
    def pad_token_id(self):
        return self['<pad>']

    @property
    def eos_token_id(self):
        return self['<eop>']
    @property
    def end_token_id(self):
        return self['</s>']

    @property
    def unk_token_id(self):
        return self['<unk>']

    @property
    def mask_token_id(self):
        return self['[MASK]']
    

    def _get_text_tokenizer(self):
        return self.text_tokenizer

    @staticmethod
    def get_blank_token(length: int):
        assert length >= 2
        return f"<|blank_{length}|>"

    @staticmethod
    def get_tab_token():
        return f"<|tab|>"

    @property
    def num_text_tokens(self):
        return self.text_tokenizer.num_tokens

    @property
    def num_tokens(self):
        return self.num_image_tokens + self.num_text_tokens

    @staticmethod
    def _encode_whitespaces(text: str, max_len: int = 80):
        text = text.replace("\t", SPTokenizer.get_tab_token())
        for i in range(max_len, 1, -1):
            text = text.replace(" " * i, SPTokenizer.get_blank_token(i))
        return text

    def _preprocess(self, text: str, linebreak=True, whitespaces=True):
        if linebreak:
            text = text.replace("\n", "<n>")
        if whitespaces:
            text = self._encode_whitespaces(text, max_len=self.max_blank_length)
        return text

    def encode(
            self, text: str, linebreak=True, whitespaces=True, add_dummy_prefix=True
    ):
        """
        @param text: Text to encode.
        @param linebreak: Whether to encode newline (\n) in text.
        @param whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        @param special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        @param add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)
        if not add_dummy_prefix:
            text = "<n>" + text
        tmp = self._get_text_tokenizer().encode(text)
        print(tmp)
        tokens = [x + self.num_image_tokens for x in tmp]
        return tokens if add_dummy_prefix else tokens[2:]

    def postprocess(self, text):
        text = text.replace("<n>", "\n")
        text = text.replace(SPTokenizer.get_tab_token(), "\t")
        for i in range(2, self.max_blank_length + 1):
            text = text.replace(self.get_blank_token(i), " " * i)
        return text

    def decode(self, text_ids) -> str:
        ids = [int(_id) - self.num_image_tokens for _id in text_ids]
        ids = [_id for _id in ids if _id >= 0]
        text = self._get_text_tokenizer().decode(ids)
        text = self.postprocess(text)
        return text

    def decode_tokens(self, tokens) -> str:
        text = self._get_text_tokenizer().convert_tokens_to_string(tokens)
        text = self.postprocess(text)
        return text

    def tokenize(
            self, text: str, linebreak=True, whitespaces=True, add_dummy_prefix=True
    ):
        """
        @param text: Text to encode.
        @param linebreak: Whether to encode newline (\n) in text.
        @param whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        @param special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        @param add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)
        if not add_dummy_prefix:
            text = "<n>" + text
        tokens = self._get_text_tokenizer().tokenize(text)
        return tokens if add_dummy_prefix else tokens[2:]

    def __getitem__(self, x):
        if isinstance(x, int):
            if x < self.num_image_tokens:
                return "<image_{}>".format(x)
            else:
                return self.text_tokenizer.convert_id_to_token(x - self.num_image_tokens)
        elif isinstance(x, str):
            if x.startswith("<image_") and x.endswith(">") and x[7:-1].isdigit():
                return int(x[7:-1])
            else:
                return self.text_tokenizer.convert_token_to_id(x) + self.num_image_tokens
        else:
            raise ValueError("The key should be str or int.")

    def convert_tokens_to_ids(self, tokens):
        return self._get_text_tokenizer().convert_tokens_to_ids(tokens)

