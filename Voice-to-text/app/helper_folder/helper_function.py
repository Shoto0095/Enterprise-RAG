import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from datetime import datetime
from ..video_to_text import transcribe_video
from .ingest_pdf import ingest_pdf
from .job_status import JOB_STATUS
from ..chatbot import restart_chatbot
from ..logger import get_logger

_logger = get_logger("helper_function")

PDF_FOLDER = os.path.join(os.getcwd(), 'PDFs')


def create_pdf_from_text(text, video_name):
    """Create PDF file from transcribed text"""
    pdf_filename = f"{os.path.splitext(video_name)[0]}.pdf"
    pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='#1f4788',
        spaceAfter=12,
        alignment=1
    )
    
    content = []
    content.append(Paragraph(f"Transcription: {os.path.splitext(video_name)[0]}", title_style))
    content.append(Spacer(1, 0.3*inch))
    content.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    content.append(Spacer(1, 0.2*inch))
    content.append(Paragraph(text, styles['Normal']))
    
    doc.build(content)
    return pdf_path



def process_video_pipeline(video_path: str, filename: str, job_id: str = None):
    """
    Full pipeline:
    video -> transcript -> PDF -> DB ingest
    """
    try:
        transcript_text = transcribe_video(video_path)
        pdf_path = create_pdf_from_text(transcript_text, filename)
        success = ingest_pdf(pdf_path)

        if not success:
            raise RuntimeError("PDF ingestion failed during video pipeline")

        if job_id and job_id in JOB_STATUS:
            JOB_STATUS[job_id]["status"] = "success"
            JOB_STATUS[job_id]["message"] = "Processing completed successfully"

        _logger.info(f"Pipeline completed for {filename}")

        # Refresh chatbot memory to pick up latest DB data
        restart_chatbot()

    except Exception as e:
        if job_id and job_id in JOB_STATUS:
            JOB_STATUS[job_id]["status"] = "failed"
            JOB_STATUS[job_id]["message"] = str(e)

        _logger.error(f" Error in processing pipeline for {filename}: {str(e)}")

