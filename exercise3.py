import nltk
import ssl
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt

# 禁用SSL证书验证
ssl._create_default_https_context = ssl._create_unverified_context

# 下载NLTK资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# 读取Moby Dick文本
with open('moby_dick.txt', 'r', encoding='utf-8') as file:
    moby_dick_text = file.read()

# 分词
tokens = word_tokenize(moby_dick_text)

# 停用词过滤
stop_words = set(stopwords.words('english'))
filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

# POS标记
pos_tags = pos_tag(filtered_tokens)

# 计算POS频率
pos_freq = FreqDist(tag for word, tag in pos_tags)
common_pos = pos_freq.most_common(5)

print("最常见的词性标记：")
for tag, count in common_pos:
    print(f"{tag}: {count}")

# 词形还原
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    # 将POS标记映射到WordNet词性
    if tag.startswith('J'):
        return 'a'  # 形容词
    elif tag.startswith('V'):
        return 'v'  # 动词
    elif tag.startswith('N'):
        return 'n'  # 名词
    elif tag.startswith('R'):
        return 'r'  # 副词
    else:
        return 'n'  # 默认将其视为名词

lemmatized_tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]

# 绘制POS频率分布图
pos_freq.plot(20, title='POS频率分布图')
plt.show()

# 情感分析
blob = TextBlob(moby_dick_text)
sentiment_score = blob.sentiment.polarity

# 确定整体情感
if sentiment_score > 0.05:
    sentiment = "积极"
elif sentiment_score < -0.05:
    sentiment = "消极"
else:
    sentiment = "中性"

print(f"平均情感分数：{sentiment_score}")
print(f"整体文本情感：{sentiment}")
