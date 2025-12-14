from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from typing import Dict
from datetime import datetime
from io import BytesIO
import os

# 日本語フォントを登録（Windows標準フォントを使用）
def register_japanese_fonts():
    """日本語フォントを登録"""
    fonts_dir = "C:/Windows/Fonts"
    
    # 游ゴシックを優先、なければメイリオを使用
    font_paths = [
        ("YuGothic", os.path.join(fonts_dir, "YuGothR.ttc")),
        ("YuGothicBold", os.path.join(fonts_dir, "YuGothB.ttc")),
        ("Meiryo", os.path.join(fonts_dir, "meiryo.ttc")),
        ("MeiryoBold", os.path.join(fonts_dir, "meiryob.ttc")),
        ("MSGothic", os.path.join(fonts_dir, "msgothic.ttc")),
    ]
    
    registered_fonts = {}
    for font_name, font_path in font_paths:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path, subfontIndex=0))
                registered_fonts[font_name] = True
            except Exception as e:
                print(f"Warning: Could not register font {font_name}: {e}")
    
    # 使用するフォントを決定
    if "YuGothic" in registered_fonts:
        return "YuGothic", "YuGothicBold" if "YuGothicBold" in registered_fonts else "YuGothic"
    elif "Meiryo" in registered_fonts:
        return "Meiryo", "MeiryoBold" if "MeiryoBold" in registered_fonts else "Meiryo"
    elif "MSGothic" in registered_fonts:
        return "MSGothic", "MSGothic"
    else:
        # フォントが見つからない場合はデフォルトを返す
        return "Helvetica", "Helvetica-Bold"

# フォントを登録
FONT_NORMAL, FONT_BOLD = register_japanese_fonts()

def generate_pdf(report: Dict) -> BytesIO:
    """レポートからPDFを生成"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                            rightMargin=30*mm, leftMargin=30*mm,
                            topMargin=30*mm, bottomMargin=30*mm)
    
    # スタイルの定義（日本語フォントを使用）
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=FONT_BOLD,
        fontSize=18,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName=FONT_BOLD,
        fontSize=14,
        textColor=colors.HexColor('#2c7fb8'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=FONT_NORMAL,
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    )
    
    # コンテンツの構築
    story = []
    
    # タイトル
    story.append(Paragraph("定期健診レポート", title_style))
    story.append(Spacer(1, 10*mm))
    
    # 基本情報
    exam_date = report.get("examination_date", "")
    if exam_date:
        try:
            dt = datetime.fromisoformat(exam_date.replace('Z', '+00:00'))
            date_str = dt.strftime("%Y年%m月%d日 %H:%M")
        except:
            date_str = exam_date
    else:
        date_str = "不明"
    
    info_data = [
        ["検査日時", date_str],
        ["セッションID", report.get("session_id", "不明")],
    ]
    
    if report.get("duration_minutes"):
        info_data.append(["通話時間", f"{report['duration_minutes']:.1f}分"])
    
    info_table = Table(info_data, colWidths=[40*mm, 130*mm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), FONT_BOLD),
        ('FONTNAME', (1, 0), (1, -1), FONT_NORMAL),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 10*mm))
    
    # サマリー
    if report.get("summary"):
        story.append(Paragraph("サマリー", heading_style))
        story.append(Paragraph(report["summary"], normal_style))
        story.append(Spacer(1, 8*mm))
    
    # 体調の変化
    if report.get("health_changes"):
        health = report["health_changes"]
        story.append(Paragraph("1. 最近の体調の変化", heading_style))
        status_text = health.get("status", "不明")
        status_color_hex = "#28a745" if status_text == "良好" else "#ffc107" if status_text == "注意" else "#dc3545"
        story.append(Paragraph(f"<b>状態:</b> <font color='{status_color_hex}'>{status_text}</font>", normal_style))
        story.append(Paragraph(health.get("details", "情報なし"), normal_style))
        story.append(Spacer(1, 8*mm))
    
    # 食事の状況
    if report.get("diet"):
        diet = report["diet"]
        story.append(Paragraph("2. 食事の状況", heading_style))
        status_text = diet.get("status", "不明")
        status_color_hex = "#28a745" if status_text == "良好" else "#ffc107" if status_text == "注意" else "#dc3545"
        story.append(Paragraph(f"<b>状態:</b> <font color='{status_color_hex}'>{status_text}</font>", normal_style))
        story.append(Paragraph(diet.get("details", "情報なし"), normal_style))
        story.append(Spacer(1, 8*mm))
    
    # 運動の状況
    if report.get("exercise"):
        exercise = report["exercise"]
        story.append(Paragraph("3. 運動の状況", heading_style))
        status_text = exercise.get("status", "不明")
        status_color_hex = "#28a745" if status_text == "良好" else "#ffc107" if status_text == "注意" else "#dc3545"
        story.append(Paragraph(f"<b>状態:</b> <font color='{status_color_hex}'>{status_text}</font>", normal_style))
        story.append(Paragraph(exercise.get("details", "情報なし"), normal_style))
        story.append(Spacer(1, 8*mm))
    
    # 薬の服用状況
    if report.get("medication"):
        medication = report["medication"]
        story.append(Paragraph("4. 薬の服用状況", heading_style))
        status_text = medication.get("status", "不明")
        status_color_hex = "#28a745" if status_text == "良好" else "#ffc107" if status_text == "注意" else "#dc3545"
        story.append(Paragraph(f"<b>状態:</b> <font color='{status_color_hex}'>{status_text}</font>", normal_style))
        story.append(Paragraph(medication.get("details", "情報なし"), normal_style))
        story.append(Spacer(1, 8*mm))
    
    # 次回の来院予定
    if report.get("next_visit"):
        next_visit = report["next_visit"]
        story.append(Paragraph("5. 次回の来院予定", heading_style))
        confirmed_text = "確認済み" if next_visit.get("confirmed", False) else "未確認"
        story.append(Paragraph(f"<b>確認状況:</b> {confirmed_text}", normal_style))
        story.append(Paragraph(next_visit.get("details", "情報なし"), normal_style))
        story.append(Spacer(1, 8*mm))
    
    # 懸念事項
    if report.get("concerns") and len(report["concerns"]) > 0:
        story.append(Paragraph("懸念事項", heading_style))
        for concern in report["concerns"]:
            story.append(Paragraph(f"• {concern}", normal_style))
        story.append(Spacer(1, 8*mm))
    
    # 推奨事項
    if report.get("recommendations") and len(report["recommendations"]) > 0:
        story.append(Paragraph("推奨事項", heading_style))
        for recommendation in report["recommendations"]:
            story.append(Paragraph(f"• {recommendation}", normal_style))
        story.append(Spacer(1, 8*mm))
    
    # PDFを生成
    doc.build(story)
    buffer.seek(0)
    return buffer
