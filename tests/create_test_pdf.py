import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

def create_test_pdf(file_path: str, title: str, text_lines: list):
    """
    Creates a simple PDF document for testing purposes.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        c = canvas.Canvas(file_path, pagesize=letter)
        width, height = letter

        # --- Set Document Title ---
        c.setTitle(title)

        # --- Write Content ---
        text = c.beginText()
        # Set font and size
        text.setFont("Helvetica", 12)
        # Place text object on the page
        text.setTextOrigin(inch, height - inch)
        text.setLeading(14) # Line spacing

        # Write title and an empty line
        text.setFont("Helvetica-Bold", 16)
        text.textLine(title)
        # moveCursor uses a positive dy to move down the page
        text.moveCursor(0, 0.2 * inch)

        # Write the body text
        text.setFont("Helvetica", 12)
        for line in text_lines:
            text.textLine(line)

        c.drawText(text)
        c.showPage()
        c.save()
        print(f"Successfully created test PDF: {file_path}")
        return True
    except Exception as e:
        print(f"Error creating test PDF: {e}")
        return False

if __name__ == "__main__":
    # Define the content for the test PDF
    PDF_FILE_PATH = os.path.join(os.path.dirname(__file__), "test_document.pdf")
    PDF_TITLE = "The Future of Artificial Intelligence"
    PDF_TEXT = [
        "Artificial Intelligence (AI) is a branch of computer science that aims to create",
        "intelligent machines. It has become an essential part of the technology industry.",
        "Research in AI is concerned with producing machines to automate tasks requiring",
        "intelligent behavior. Examples include control, planning and scheduling,",
        "the ability to answer diagnostic and consumer questions, and handwriting recognition.",
        "",
        "The field was founded on the claim that a central property of humans, intelligence—",
        "the sapience of Homo sapiens—can be so precisely described that it can be simulated",
        "by a machine. This raises philosophical issues about the nature of the mind and the",
        "ethics of creating artificial beings endowed with human-like intelligence.",
    ]

    # Create the PDF
    create_test_pdf(PDF_FILE_PATH, PDF_TITLE, PDF_TEXT)

    # --- Create a second, empty PDF for testing edge cases ---
    EMPTY_PDF_PATH = os.path.join(os.path.dirname(__file__), "empty_document.pdf")
    try:
        c = canvas.Canvas(EMPTY_PDF_PATH, pagesize=letter)
        c.showPage()
        c.save()
        print(f"Successfully created empty test PDF: {EMPTY_PDF_PATH}")
    except Exception as e:
        print(f"Error creating empty test PDF: {e}")

    # --- Create a third, image-only PDF (simulated) ---
    # We can't add real images, but we can create a PDF with no extractable text
    IMAGE_PDF_PATH = os.path.join(os.path.dirname(__file__), "image_only_document.pdf")
    try:
        c = canvas.Canvas(IMAGE_PDF_PATH, pagesize=letter)
        # Draw a shape instead of text
        c.drawString(72, 72, "") # Empty string to ensure no text is extracted
        c.showPage()
        c.save()
        print(f"Successfully created image-only test PDF: {IMAGE_PDF_PATH}")
    except Exception as e:
        print(f"Error creating image-only test PDF: {e}")