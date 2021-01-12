# 데이터 로드와 사전처리 데이터 : TF. 텍스트
# TensorFlow Text는 TensorFlow 2.0에서 사용할 준비가 된 텍스트 관련 클래스 및 작업 모음을 제공
# 기반 모델에 필요한 사전 처리를 정기적으로 수행 할 수 있으며 
# 핵심 TensorFlow에서 제공하지 않는 시퀀스 모델링에 유용한 기타 기능을 포함
# 목차
# 1. 열렬한 실행
# 2. 유니코드
# 3. 토큰화
# 4. 기타 텍스트 작업


# 1. 열렬한 실행
# TensorFlow Text에는 TensorFlow 2.0이 필요하며 eager 모드 및 그래프 모드와 완벽하게 호환됩
import tensorflow as tf
import tensorflow_text as text

# 2. 유니 코드
# 대부분의 작업은 문자열이 UTF-8로되어 있다고 예상합니다. 
# 다른 인코딩을 사용하는 경우 핵심 tensorflow 트랜스 코딩 작업을 사용하여 UTF-8로 트랜스 코딩 할 수 있습니다. 
# 입력이 유효하지 않은 경우 동일한 op를 사용하여 문자열을 구조적으로 유효한 UTF-8로 강제 변환 할 수도 있습니다.

docs = tf.constant([u'Everything not saved will be lost.'.encode('UTF-16-BE'), u'Sad☹'.encode('UTF-16-BE')])
utf8_docs = tf.strings.unicode_transcode(docs, input_encoding='UTF-16-BE', output_encoding='UTF-8')

# 3. 토큰 화
# WhitespaceTokenizer : ICU 정의 공백 문자 (예 : 공백, 탭, 새 줄)에서 UTF-8 문자열을 분할하는 기본 토크 나이저
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())

# UnicodeScriptTokenizer : 이 토크 나이 저는 유니 코드 스크립트 경계를 기반으로 UTF-8 문자열을 분할합니다. 
# 사용되는 스크립트 코드는 ICU (International Components for Unicode) UScriptCode 값에 해당합니다. 
# 참조 : http://icu-project.org/apiref/icu4c/uscript_8h.html
# 실제로 이것은 WhitespaceTokenizer 와 유사하지만 언어 텍스트 (예 : USCRIPT_LATIN, USCRIPT_CYRILLIC 등)에서 
# 구두점 (USCRIPT_COMMON)을 분리하는 동시에 언어 텍스트를 서로 분리한다는 점이 가장 뚜렷합니다.
tokenizer = text.UnicodeScriptTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())

# 유니 코드 분할 : 공백없이 언어를 토큰 화하여 단어를 분할 할 때 문자별로 분할하는 것이 일반적이며 
# 코어에있는 unicode_split op를 사용하여 수행 할 수 있습니다.
tokens = tf.strings.unicode_split([u"仅今年前".encode('UTF-8')], 'UTF-8')
print(tokens.to_list())

# 오프셋 : 문자열을 토큰화할 때 원래 문자열에서 토큰의 출처를 알고 싶은 경우가 종종 있습니다. 
# 이러한 이유로 TokenizerWithOffsets 를 구현하는 각 토크 나이저에는 
# 토큰과 함께 바이트 오프셋을 반환하는 tokenize_with_offsets 메서드가 있습니다. 
# offset_starts는 각 토큰이 시작하는 원래 문자열의 바이트를 나열하고 offset_limits는 각 토큰이 끝나는 바이트를 나열합니다.
tokenizer = text.UnicodeScriptTokenizer()
(tokens, offset_starts, offset_limits) = tokenizer.tokenize_with_offsets(['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())
print(offset_starts.to_list())
print(offset_limits.to_list())

# TF.Data 예 : Tokenizer는 tf.data API에서 예상대로 작동합니다. 아래에 간단한 예가 나와 있습니다.
docs = tf.data.Dataset.from_tensor_slices([['Never tell me the odds.'], ["It's a trap!"]])
tokenizer = text.WhitespaceTokenizer()
tokenized_docs = docs.map(lambda x: tokenizer.tokenize(x))
iterator = iter(tokenized_docs)
print(next(iterator).to_list())
print(next(iterator).to_list())

# 기타 텍스트 작업 : TF.Text는 다른 유용한 전처리 작업을 패키지합니다. 아래에서 몇 가지를 검토하겠습니다.
# 워드 셰이프 : 일부 자연어 이해 모델에서 사용되는 일반적인 기능은 텍스트 문자열에 특정 속성이 있는지 확인하는 것입니다. 
# 예를 들어, 문장 분리 모델에는 단어 대문자 사용 또는 구두점 문자가 문자열 끝에 있는지 확인하는 기능이 포함될 수 있습니다.
# Wordshape는 입력 텍스트에서 다양한 관련 패턴을 일치시키기 위해 유용한 정규식 기반 도우미 함수를 다양하게 정의합니다. 
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['Everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])

# Is capitalized?
f1 = text.wordshape(tokens, text.WordShape.HAS_TITLE_CASE)
# Are all letters uppercased?
f2 = text.wordshape(tokens, text.WordShape.IS_UPPERCASE)
# Does the token contain punctuation?
f3 = text.wordshape(tokens, text.WordShape.HAS_SOME_PUNCT_OR_SYMBOL)
# Is the token a number?
f4 = text.wordshape(tokens, text.WordShape.IS_NUMERIC_VALUE)

print(f1.to_list())
print(f2.to_list())
print(f3.to_list())
print(f4.to_list())

# N- 그램 및 슬라이딩 윈도우 :  N- 그램은 슬라이딩 윈도우 크기가 n 인 경우 순차적 인 단어입니다. 
# 토큰을 결합 할 때 지원되는 세 가지 감소 메커니즘이 있습니다. 
# 텍스트의 경우 문자열을 서로 추가하는 Reduction.STRING_JOIN 을 사용할 수 있습니다. 
# 기본 구분 문자는 공백이지만 string_separater 인수로 변경할 수 있습니다.
# 다른 두 가지 감소 방법은 숫자 값과 함께 가장 자주 사용되며 Reduction.SUM 및 Reduction.MEAN 입니다.
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['Everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])

# Ngrams, in this case bi-gram (n = 2)
bigrams = text.ngrams(tokens, 2, reduction_type=text.Reduction.STRING_JOIN)

print(bigrams.to_list())

