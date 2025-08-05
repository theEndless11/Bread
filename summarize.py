from transformers import pipeline
import re

# Load summarizer once
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

def chunk_text(text, max_words=600):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_length + word_count > max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = word_count
        else:
            current_chunk.append(sentence)
            current_length += word_count
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def truncate_to_words(text, max_words=30):
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = words[:max_words]
    # Try to end at a sentence boundary if possible
    for i in reversed(range(len(truncated))):
        if truncated[i].endswith(('.', '!', '?')):
            return ' '.join(truncated[:i+1])
    # Otherwise, just return truncated with period
    return ' '.join(truncated).rstrip('.,;:') + '.'

def scale_summary_length(word_count):
    if word_count < 50:
        return (20, 30)
    elif word_count < 150:
        return (30, 50)
    elif word_count < 500:
        return (50, 100)
    else:
        return (80, 160)

def summarize_text(text):
    word_count = len(text.split())
    min_len, max_len = scale_summary_length(word_count)

    def run_summary(text_input, max_len_param):
        return summarizer(text_input,
                          max_length=max_len_param,
                          min_length=min_len,
                          do_sample=False)[0]['summary_text']

    if word_count <= 500:
        summary = run_summary(text, max_len)
        # Retry with higher max_length if too short or ends abruptly
        if len(summary.split()) < (min_len // 2) or not summary.strip().endswith('.'):
            summary = run_summary(text, max_len + 30)
        return summary
    else:
        # For long texts, do hierarchical summarization
        chunks = chunk_text(text, max_words=600)
        first_pass_min = max(15, min_len // 3)
        first_pass_max = max(30, max_len // 3)

        chunk_summaries = []
        for chunk in chunks:
            chunk_summary = summarizer(chunk,
                                      max_length=first_pass_max,
                                      min_length=first_pass_min,
                                      do_sample=False)[0]['summary_text']
            chunk_summaries.append(chunk_summary)

        combined_summary = " ".join(chunk_summaries)
        final_summary = summarizer(combined_summary,
                                  max_length=max_len,
                                  min_length=min_len,
                                  do_sample=False)[0]['summary_text']
        # Retry if too short or incomplete
        if len(final_summary.split()) < (min_len // 2) or not final_summary.strip().endswith('.'):
            final_summary = summarizer(combined_summary,
                                      max_length=max_len + 30,
                                      min_length=min_len,
                                      do_sample=False)[0]['summary_text']
        return final_summary

def improve_summary(summary, max_words=30):
    # Clean-up repetitive or awkward phrases (expand as needed)
    summary = re.sub(r'(\bthis is a good example of\b.*?\. )', '', summary, flags=re.I)
    summary = re.sub(r'\b(\w+)( \1\b)+', r'\1', summary)
    summary = re.sub(r'(\b\w+\b)( and \1\b)+', r'\1', summary, flags=re.I)
    summary = re.sub(r'\s+([.,])', r'\1', summary)
    summary = re.sub(r'\. ([Tt]he)', r'; \1', summary)
    summary = re.sub(r'\s{2,}', ' ', summary)
    summary = summary.strip()
    if summary:
        summary = summary[0].upper() + summary[1:]
    if summary and not summary.endswith('.'):
        summary += '.'

    # Dynamic truncation based on input length
    max_summary_words = max(20, min(100, max_words))
    summary = truncate_to_words(summary, max_words=max_summary_words)
    return summary.strip()

if __name__ == "__main__":
    print("Enter your text to summarize (press Enter twice to submit). Type 'exit' to quit.")
    while True:
        lines = []
        while True:
            line = input()
            if line.strip().lower() == "exit":
                print("Exiting...")
                exit(0)
            if line.strip() == "":
                break
            lines.append(line)
        user_text = " ".join(lines).strip()

        if not user_text:
            print("No input provided. Please enter text or type 'exit' to quit.")
            continue

        raw_summary = summarize_text(user_text)
        max_words = min(100, max(30, int(len(user_text.split()) * 0.3)))
        final_summary = improve_summary(raw_summary, max_words=max_words)
        print("\n Summary:\n", final_summary)
        print("\nEnter next text to summarize (or 'exit' to quit):")
