import nltk
import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.tokenizers import Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    st.header("Text Summarization App")
    text = st.text_area(label="Input text")


    def sentiment_scores(sentence):
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(sentence)
        emotion = ""
        if sentiment_dict['compound'] >= 0.05:
            emotion = "Positive"
        elif sentiment_dict['compound'] <= - 0.05:
            emotion = "Negative"
        else:
            emotion = "Neutral"
        return emotion
    if st.button("Summarize"):
        if text:
            summarizer_kl = KLSummarizer()
            language = "english"
            sentence_count = 5

            parser = PlaintextParser(text, Tokenizer(language))

            # Summarize using sumy KL Divergence
            summary = summarizer_kl(parser.document, 2)

            kl_summary = ""
            for sentence in summary:
                kl_summary += str(sentence)



            st.success(kl_summary)
    if st.button("Sentiment"):
        if text:
            summarizer_kl = KLSummarizer()
            language = "english"
            sentence_count = 5

            parser = PlaintextParser(text, Tokenizer(language))

            # Summarize using sumy KL Divergence
            summary = summarizer_kl(parser.document, 2)

            kl_summary = ""
            for sentence in summary:
                kl_summary += str(sentence)

                score = sentiment_scores(kl_summary)
            st.success(score)







