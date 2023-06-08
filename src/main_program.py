from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from functions import *
from photo_scanning import *
import sys
import re

from predict import *

import os
import uuid

FONT_SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 18, 24, 36, 48, 64, 72, 96, 144, 288]
IMAGE_EXTENSIONS = ['.jpg','.png','.bmp']
HTML_EXTENSIONS = ['.htm', '.html']

def hexuuid():
    return uuid.uuid4().hex

def splitext(p):
    return os.path.splitext(p)[1].lower()

class TextEdit(QTextEdit):

    def canInsertFromMimeData(self, source):

        if source.hasImage():
            return True
        else:
            return super(TextEdit, self).canInsertFromMimeData(source)

    def insertFromMimeData(self, source):

        cursor = self.textCursor()
        document = self.document()

        if source.hasUrls():

            for u in source.urls():
                file_ext = splitext(str(u.toLocalFile()))
                if u.isLocalFile() and file_ext in IMAGE_EXTENSIONS:
                    image = QImage(u.toLocalFile())
                    document.addResource(QTextDocument.ImageResource, u, image)
                    cursor.insertImage(u.toLocalFile())

                else:
                    break

            else:
                return


        elif source.hasImage():
            image = source.imageData()
            uuid = hexuuid()
            document.addResource(QTextDocument.ImageResource, uuid, image)
            cursor.insertImage(uuid)
            return

        super(TextEdit, self).insertFromMimeData(source)


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        title = "Система опрацювання тексту"
        
        layout = QVBoxLayout()
        self.editor = TextEdit()
        self.editor1 = TextEdit()
        self.editor.setAutoFormatting(QTextEdit.AutoAll)
        self.editor.selectionChanged.connect(self.update_format)
        font = QFont('Times', 12)
        self.editor.setFont(font)
        self.editor1.setFontPointSize(12)

        self.editor1.setAutoFormatting(QTextEdit.AutoAll)
        self.editor1.selectionChanged.connect(self.update_format)
        self.editor1.setFont(font)
        self.editor1.setFontPointSize(12)

        self.path = None
        label1 = QLabel("Оригінальний текст:", self)
        label2 = QLabel("Результат:", self)
        layout.addWidget(label1)
        layout.addWidget(self.editor)
        layout.addWidget(label2)
        layout.addWidget(self.editor1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        file_toolbar = QToolBar("Файл")
        file_toolbar.setIconSize(QSize(14, 14))
        self.addToolBar(file_toolbar)
        file_menu = self.menuBar().addMenu("&Файл")

        open_file_action = QAction(QIcon(os.path.join('images', 'blue-folder-open-document.png')), "Відкрити файл...", self)
        open_file_action.setStatusTip("Відкрити файл")
        open_file_action.triggered.connect(self.file_open)
        file_menu.addAction(open_file_action)
        file_toolbar.addAction(open_file_action)

        save_file_action = QAction(QIcon(os.path.join('images', 'disk.png')), "Зберегти", self)
        save_file_action.setStatusTip("Зберегти поточну сторінку")
        save_file_action.triggered.connect(self.file_save)
        file_menu.addAction(save_file_action)
        file_toolbar.addAction(save_file_action)

        saveas_file_action = QAction(QIcon(os.path.join('images', 'disk--pencil.png')), "Зберегти як...", self)
        saveas_file_action.setStatusTip("Зберегти поточну сторінку у файлі")
        saveas_file_action.triggered.connect(self.file_saveas)
        file_menu.addAction(saveas_file_action)
        file_toolbar.addAction(saveas_file_action)

        print_action = QAction(QIcon(os.path.join('images', 'printer.png')), "Друк...", self)
        print_action.setStatusTip("Друк поточної сторінки")
        print_action.triggered.connect(self.file_print)
        file_menu.addAction(print_action)
        file_toolbar.addAction(print_action)

        edit_toolbar = QToolBar("Редагувати")
        edit_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(edit_toolbar)
        edit_menu = self.menuBar().addMenu("&Редагувати")

        undo_action = QAction(QIcon(os.path.join('images', 'arrow-curve-180-left.png')), "Відмнити", self)
        undo_action.setStatusTip("Відмінити останню дію")
        undo_action.triggered.connect(self.editor.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction(QIcon(os.path.join('images', 'arrow-curve.png')), "Відновити", self)
        redo_action.setStatusTip("Відновити останню дію")
        redo_action.triggered.connect(self.editor.redo)
        edit_toolbar.addAction(redo_action)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        cut_action = QAction(QIcon(os.path.join('images', 'scissors.png')), "Вирізати", self)
        cut_action.setStatusTip("Вирізати виділений текст")
        cut_action.setShortcut(QKeySequence.Cut)
        cut_action.triggered.connect(self.editor.cut)
        edit_toolbar.addAction(cut_action)
        edit_menu.addAction(cut_action)

        copy_action = QAction(QIcon(os.path.join('images', 'document-copy.png')), "Скопіювати", self)
        copy_action.setStatusTip("Скопіювати виділений текст")
        cut_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.editor.copy)
        edit_toolbar.addAction(copy_action)
        edit_menu.addAction(copy_action)

        paste_action = QAction(QIcon(os.path.join('images', 'clipboard-paste-document-text.png')), "Вставити", self)
        paste_action.setStatusTip("Вставити")
        cut_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(self.editor.paste)
        edit_toolbar.addAction(paste_action)
        edit_menu.addAction(paste_action)

        select_action = QAction(QIcon(os.path.join('images', 'selection-input.png')), "Виділити все", self)
        select_action.setStatusTip("Виділити увесь текст")
        cut_action.setShortcut(QKeySequence.SelectAll)
        select_action.triggered.connect(self.editor.selectAll)
        edit_menu.addAction(select_action)

        edit_menu.addSeparator()

        format_toolbar = QToolBar("Форматування")
        format_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(format_toolbar)
        format_menu = self.menuBar().addMenu("&Форматування")

        text_toolbar = QToolBar("Аналіз тексту")
        text_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(text_toolbar)
        text_menu = self.menuBar().addMenu("&Аналіз тексту")

        self.fonts = QFontComboBox()
        self.fonts.currentFontChanged.connect(self.editor.setCurrentFont)
        format_toolbar.addWidget(self.fonts)

        self.fontsize = QComboBox()
        self.fontsize.addItems([str(s) for s in FONT_SIZES])

        self.fontsize.currentIndexChanged[str].connect(lambda s: self.editor.setFontPointSize(float(s)) )
        format_toolbar.addWidget(self.fontsize)

        self.bold_action = QAction(QIcon(os.path.join('images', 'edit-bold.png')), "Жирний", self)
        self.bold_action.setStatusTip("Жирний")
        self.bold_action.setShortcut(QKeySequence.Bold)
        self.bold_action.setCheckable(True)
        self.bold_action.toggled.connect(lambda x: self.editor.setFontWeight(QFont.Bold if x else QFont.Normal))
        format_toolbar.addAction(self.bold_action)
        format_menu.addAction(self.bold_action)

        self.italic_action = QAction(QIcon(os.path.join('images', 'edit-italic.png')), "Курсив", self)
        self.italic_action.setStatusTip("Курсив")
        self.italic_action.setShortcut(QKeySequence.Italic)
        self.italic_action.setCheckable(True)
        self.italic_action.toggled.connect(self.editor.setFontItalic)
        format_toolbar.addAction(self.italic_action)
        format_menu.addAction(self.italic_action)

        self.underline_action = QAction(QIcon(os.path.join('images', 'edit-underline.png')), "Підкреслення", self)
        self.underline_action.setStatusTip("Підкреслення")
        self.underline_action.setShortcut(QKeySequence.Underline)
        self.underline_action.setCheckable(True)
        self.underline_action.toggled.connect(self.editor.setFontUnderline)
        format_toolbar.addAction(self.underline_action)
        format_menu.addAction(self.underline_action)

        format_menu.addSeparator()

        self.alignl_action = QAction(QIcon(os.path.join('images', 'edit-alignment.png')), "Вирівняти по лівій стороні", self)
        self.alignl_action.setStatusTip("Вирівняти по лівій стороні")
        self.alignl_action.setCheckable(True)
        self.alignl_action.triggered.connect(lambda: self.editor.setAlignment(Qt.AlignLeft))
        format_toolbar.addAction(self.alignl_action)
        format_menu.addAction(self.alignl_action)

        self.alignc_action = QAction(QIcon(os.path.join('images', 'edit-alignment-center.png')), "Вирівняти по центру", self)
        self.alignc_action.setStatusTip("Вирівняти по центру")
        self.alignc_action.setCheckable(True)
        self.alignc_action.triggered.connect(lambda: self.editor.setAlignment(Qt.AlignCenter))
        format_toolbar.addAction(self.alignc_action)
        format_menu.addAction(self.alignc_action)

        self.alignr_action = QAction(QIcon(os.path.join('images', 'edit-alignment-right.png')), "Вирівняти по правій стороні", self)
        self.alignr_action.setStatusTip("Вирівняти по правій стороні")
        self.alignr_action.setCheckable(True)
        self.alignr_action.triggered.connect(lambda: self.editor.setAlignment(Qt.AlignRight))
        format_toolbar.addAction(self.alignr_action)
        format_menu.addAction(self.alignr_action)

        self.key_extraction = QAction("Виявлення ключових слів", self)

        self.key_extraction.triggered.connect(lambda: self.editor1.setText(key_words_extraction(str(self.editor.toPlainText()))))
        text_menu.addAction(self.key_extraction)

        self.name_entity_recognition = QAction("Розпізнавання іменованих сутностей", self)
        self.name_entity_recognition.triggered.connect(lambda: self.editor1.setHtml(self.get_entities()))
        text_menu.addAction(self.name_entity_recognition)
        format_group = QActionGroup(self)
        format_group.setExclusive(True)
        format_group.addAction(self.alignl_action)
        format_group.addAction(self.alignc_action)
        format_group.addAction(self.alignr_action)

        format_menu.addSeparator()

        self._format_actions = [
            self.fonts,
            self.fontsize,
            self.bold_action,
            self.italic_action,
            self.underline_action,
        ]
        self.update_format()
        self.update_title()
        self.setWindowTitle(title)
        self.show()

    def get_entities(self):
        dicti = find_named_entities(str(self.editor.toPlainText()))
        print(dicti)
        tag_colors = {'B-tim': '#FFD4B2', 'I-tim': '#FFD4B2', 'B-gpe': '#FFF6BD', 'I-gpe': '#FFF6BD',
        'B-geo': '#E3ACF9', 'I-geo': '#E3ACF9', 'B-per': '#86C8BC', 'I-per': '#86C8BC', 'I-org': 'pink', 'B-org': 'pink'}
        dict_labels = {'Time': '#FFD4B2', 'Geopolitical term': '#FFF6BD', 'Location': '#E3ACF9', 'Person':'#86C8BC', 'Organisation': 'pink' }
        result_string = ""
        token_string = str(self.editor.toPlainText()).split()
        array = []
        for word in token_string:
            word1 = re.sub("[^A-Za-z0-9 ]","",word)
            if dicti.get(word1) in tag_colors:
                array.append(tag_colors[dicti[word1]])
            elif dicti.get(word) in tag_colors:
                array.append(tag_colors[dicti[word]])
            else:
                array.append('O')
        s = []
        w = ''
        res = []
        for i in range(len(array)):
            if len(s) != 0:
                if array[i] != 'O':
                    if s[-1] == array[i]:
                        w += token_string[i] + " "
                        s.append(array[i])
                        res.append((w, array[i]))
                        w = ""
                        s = []
                    else:
                        w = token_string[i] + " "
                        s = [array[i]]
                else:
                    res.append((w, s[-1]))
                    w = ''
                    s = []
                    res.append((token_string[i], 'O'))
            else:
                if array[i] != 'O':
                    w = token_string[i] + " "
                    s = [array[i]] 
                else:
                    res.append((token_string[i], 'O'))
        if len(s) != 0:
            res.append((w, array[len(array) - 1]))

        result_string = ""
        for key in dict_labels.keys():
            result_string += f'<span style="border-radius: 20px; background-color:{dict_labels[key]};display: inline">{key}</span> '
        result_string += '<hr>'
        for r in res:
            if r[1] != 'O':
                result_string += f'<span style="border-radius: 20px; background-color:{r[1]}; display: inline">{r[0]}</span> '
            else:
                result_string += f'<span style="display: inline">{r[0]}</span> '
        
        return result_string

    def block_signals(self, objects, b):
        for o in objects:
            o.blockSignals(b)

    def update_format(self):
     
        self.block_signals(self._format_actions, True)

        self.fonts.setCurrentFont(self.editor.currentFont())
        self.fontsize.setCurrentText(str(int(self.editor.fontPointSize())))

        self.italic_action.setChecked(self.editor.fontItalic())
        self.underline_action.setChecked(self.editor.fontUnderline())
        self.bold_action.setChecked(self.editor.fontWeight() == QFont.Bold)

        self.alignl_action.setChecked(self.editor.alignment() == Qt.AlignLeft)
        self.alignc_action.setChecked(self.editor.alignment() == Qt.AlignCenter)
        self.alignr_action.setChecked(self.editor.alignment() == Qt.AlignRight)

        self.block_signals(self._format_actions, False)

    def dialog_critical(self, s):
        dlg = QMessageBox(self)
        dlg.setText(s)
        dlg.setIcon(QMessageBox.Critical)
        dlg.show()

    def file_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Відкрити файл", "", "HTML documents (*.html);Text documents (*.txt);All files (*.*)")

        try:
            if '.jpg' not in path:
                with open(path, 'rU') as f:
                    text = f.read()

        except Exception as e:
            self.dialog_critical(str(e))

        else:
            self.path = path
            if '.jpg' not in self.path:
                self.editor.setText(text)
                self.update_title()
            else:
                text = main_func(self.path)
                self.editor.setText(text)
                self.update_title()

    def file_save(self):
        if self.path is None:
            return self.file_saveas()

        text = self.editor.toHtml() if splitext(self.path) in HTML_EXTENSIONS else self.editor.toPlainText()

        try:
            with open(self.path, 'w') as f:
                f.write(text)

        except Exception as e:
            self.dialog_critical(str(e))

    def file_saveas(self):
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти файл", "", "HTML documents (*.html);Text documents (*.txt);All files (*.*)")

        if not path:
            return

        text = self.editor.toHtml() if splitext(path) in HTML_EXTENSIONS else self.editor.toPlainText()

        try:
            with open(path, 'w') as f:
                f.write(text)

        except Exception as e:
            self.dialog_critical(str(e))

        else:
            self.path = path
            self.update_title()

    def file_print(self):
        dlg = QPrintDialog()
        if dlg.exec_():
            self.editor.print_(dlg.printer())

    def update_title(self):
        self.setWindowTitle("%s" % (os.path.basename(self.path) if self.path else "Untitled"))

    def edit_toggle_wrap(self):
        self.editor.setLineWrapMode( 1 if self.editor.lineWrapMode() == 0 else 0 )


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setApplicationName("Система опрацювання тексту")

    window = MainWindow()
    app.exec_()
