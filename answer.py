import logging
import re
from typing import List
from transformers import pipeline
import warnings
from dataclasses import dataclass
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class Answer:
    """Data class for answer structure."""
    answer: str
    confidence: float


class StreamlinedQASystem:
    """
    Streamlined Question-Answering System with fluent, natural-sounding answers.
    """

    def __init__(self, model_name: str = "deepset/minilm-uncased-squad2"):
        self.model_name = model_name
        self.qa_pipeline = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the QA pipeline."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=False,
                device=-1  # Use CPU
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _build_fluent_answer(self, question: str, raw_answer: str, context: str) -> str:
        """Make answer sound fluent and descriptive based on question."""
        q = question.lower()
        ans = raw_answer.strip()
        if not ans:
            return "Sorry, I couldn't find an answer."

        # Capitalize first letter & ensure ending punctuation
        ans = ans[0].upper() + ans[1:]
        if not ans.endswith(('.', '!', '?')):
            ans += '.'

        # Handle born/birth related questions for smooth phrasing
        if "born" in q or "birth" in q:
            # If answer does not start with 'in' or similar, prepend 'born in'
            if not ans.lower().startswith(("in ", "on ", "at ")):
                return f"She was born in {ans[0].lower() + ans[1:]}" if ans.endswith('.') else f"She was born in {ans[0].lower() + ans[1:]}."
            else:
                return f"She was born {ans[0].lower() + ans[1:]}" if ans.endswith('.') else f"She was born {ans[0].lower() + ans[1:]}."

        # Generic polite preambles
        preambles = [
            "She was",
            "The answer is",
            "It is",
            "This is",
            "According to the context, it is",
            "From the information, it appears to be",
        ]

        # Pick preamble based on question start
        if q.startswith(("who", "where", "when", "what", "which", "whom", "whose")):
            preamble = random.choice(preambles[:3])
            # Avoid awkward phrasing if answer already starts with article
            if preamble.endswith("was") and ans.lower().startswith(("a ", "an ", "the ")):
                preamble = preamble.replace(" was", "")
        else:
            preamble = random.choice(preambles)

        # Compose final answer
        # Avoid repetition if answer already starts similarly
        if ans.lower().startswith(preamble.lower().replace("she was", "").strip()):
            final = ans
        else:
            final = f"{preamble} {ans[0].lower() + ans[1:]}"
            final = final[0].upper() + final[1:]

        return final

    def answer_question(self, question: str, context: str) -> Answer:
        """Answer a question using the QA pipeline with fluent formatting."""
        if not question.strip() or not context.strip():
            raise ValueError("Question and context cannot be empty.")
        try:
            result = self.qa_pipeline(question=question, context=context)
            raw_answer = result['answer']
            confidence = round(result['score'], 4)
            fluent_answer = self._build_fluent_answer(question, raw_answer, context)
            return Answer(answer=fluent_answer, confidence=confidence)
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return Answer(answer="Unable to determine answer.", confidence=0.0)

    def interactive_session(self, context: str) -> None:
        """Interactive QA session."""
        print("🧠 MINI QA SYSTEM")
        print("=" * 40)
        print(f"📖 Context loaded: {len(context)} characters")
        print("💡 Ask your questions! Type 'quit' to exit")
        print("-" * 40)

        while True:
            try:
                user_input = input("\n🔍 Question: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                if not user_input:
                    print("❓ Please enter a question.")
                    continue
                result = self.answer_question(user_input, context)
                print(f"📝 Answer: {result.answer}")
                print(f"📊 Confidence: {result.confidence:.2%}")
            except KeyboardInterrupt:
                print("\n👋 Session ended.")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                logger.error(f"Interactive session error: {str(e)}")


def main():
    """Run the QA system interactively."""
    marie_curie_context = """
    Marie Curie was a Polish physicist and chemist who conducted pioneering research on radioactivity.
    She was the first woman to win a Nobel Prize and remains the only person to win Nobel Prizes in two different sciences.
    Born in Poland in 1867, Marie Curie moved to Paris to study at the University of Paris.
    Her first Nobel Prize was in Physics in 1903, shared with her husband Pierre Curie and Henri Becquerel for their work on radioactivity.
    Her second Nobel Prize was in Chemistry in 1911 for her discovery of the elements radium and polonium.
    Marie Curie died in 1934 from aplastic anemia, likely caused by exposure to radiation during her research.
    She founded the Radium Institute in Paris and Warsaw, which remain major cancer research centers today.
    During World War I, Marie Curie developed mobile X-ray units called "petites Curies" to help battlefield surgeons.
    She personally drove these units to the front lines, helping to save countless lives through improved medical imaging.
    Her daughter Irène Joliot-Curie also became a Nobel Prize winner, continuing the family's scientific legacy.
    Marie Curie's laboratory notebooks are still radioactive today and will remain so for another 1,500 years.
    She was the first female professor at the University of Paris and broke numerous barriers for women in science.
    Her work laid the foundation for modern atomic physics and led to the development of cancer treatments using radiation.
    The element curium (Cm) was named in honor of Marie and Pierre Curie for their contributions to science.
    """

    try:
        print("🚀 INITIALIZING QA SYSTEM")
        print("=" * 50)
        qa_system = StreamlinedQASystem()
        qa_system.interactive_session(marie_curie_context)
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        print(f"❌ Failed to initialize: {str(e)}")


if __name__ == "__main__":
    main()
