from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.gui_bridge import GuiBridge


class MainWindow(QMainWindow):
    def __init__(self, gui_q):
        super().__init__()
        self.setWindowTitle("Shira GUI")
        self.setMinimumSize(1280, 720)

        self._bridge = GuiBridge(gui_q, parent=self)
        self._build_ui()
        self._connect_signals()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout_main = QHBoxLayout(root)
        layout_main.setContentsMargins(0, 0, 0, 0)

        # -- ASR ---
        root_asr = QWidget()
        layout_asr = QVBoxLayout(root_asr)
        background_asr = QFrame()
        background_asr.setFrameShape(QFrame.StyledPanel)
        background_asr.setStyleSheet("background: #1a1a1a; border-radius: 8px;")
        layout_asr.addWidget(background_asr)

        layout_asr_box = QVBoxLayout(background_asr)

        # ASR stats label
        layout_asr_box.addSpacing(5)
        self.asr_stats_label = QLabel("avg: 0.00s | min: 0.00s | max: 0.00s")
        self.asr_stats_label.setStyleSheet("color: #888; font-size: 12px;")
        layout_asr_box.addWidget(self.asr_stats_label, alignment=Qt.AlignCenter)

        # Mic status pill
        layout_asr_box.addStretch(3)
        self.mic_label = QLabel("● Idle")
        self.mic_label.setStyleSheet("color: gray; font-weight: 500;")
        layout_asr_box.addWidget(self.mic_label, alignment=Qt.AlignCenter)
        layout_asr_box.addSpacing(20)

        # Transcribe status pill
        self.transcribe_label = QLabel("○ Not transcribing")
        self.transcribe_label.setStyleSheet("color: gray; font-weight: 500;")
        layout_asr_box.addWidget(self.transcribe_label, alignment=Qt.AlignCenter)
        layout_asr_box.addStretch(5)

        # Transcript view
        self.script_view = QTextEdit()
        self.script_view.setReadOnly(True)
        self.script_view.setFont(QFont("Segoe UI", 11))
        # Soft background so it's visually distinct from the window
        self.script_view.setStyleSheet(
            "QTextEdit { background: #2a2a2a; border-radius: 8px; padding: 8px; }"
        )

        # Volume bar
        self.vol_bar = QProgressBar()
        self.vol_bar.setRange(0, 100)
        self.vol_bar.setValue(0)
        self.vol_bar.setTextVisible(False)
        self.vol_bar.setFixedHeight(20)
        self.vol_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        self.vol_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #555; border-radius: 10px; background: #222; }
            QProgressBar::chunk { background: #4caf50; border-radius: 8px; }
        """)

        container_vol_bar = QWidget()
        layout_vol_bar = QHBoxLayout(container_vol_bar)
        layout_vol_bar.setContentsMargins(0, 0, 0, 30)
        layout_vol_bar.addStretch(1)
        layout_vol_bar.addWidget(self.vol_bar, stretch=6)
        layout_vol_bar.addStretch(1)

        layout_asr_box.addStretch(10)
        layout_asr_box.addWidget(container_vol_bar)


        # -- LLM ---
        root_llm = QWidget()
        layout_llm = QVBoxLayout(root_llm)
        layout_llm.addWidget(self._make_status_bar())
        layout_llm.addWidget(self._make_conversation_view(), stretch=1)

        # -- TTS ---
        root_tts = QWidget()
        layout_tts = QVBoxLayout(root_tts)
        background_tts = QFrame()
        background_tts.setFrameShape(QFrame.StyledPanel)
        background_tts.setStyleSheet("background: #1a1a1a; border-radius: 8px;")
        layout_tts.addWidget(background_tts)

        layout_tts_box = QVBoxLayout(background_tts)
        layout_tts_box.addSpacing(5)

        # TTS stats label
        self.tts_stats_label = QLabel("avg: 0.00s | min: 0.00s | max: 0.00s")
        self.tts_stats_label.setStyleSheet("color: #888; font-size: 12px;")
        layout_tts_box.addWidget(self.tts_stats_label, alignment=Qt.AlignCenter)
        layout_tts_box.addStretch()



        layout_main.addWidget(root_asr, stretch=1)
        layout_main.addWidget(root_llm, stretch=3)
        layout_main.addWidget(root_tts, stretch=1)


    
    def _make_status_bar(self):
        bar = QFrame()
        bar.setObjectName("llmStatusBar")
        bar.setStyleSheet("""
            QFrame#llmStatusBar {
                background-color: #1a1a1a;
                border-radius: 8px;
            }
        """)

        main_layout = QVBoxLayout(bar)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # -- Top row: LLM status --
        status_row = QHBoxLayout()
        status_row.addStretch()

        self.llm_label = QLabel("○ Ready")
        self.llm_label.setStyleSheet(
            "color: gray; font-weight: 500; font-size: 12px;"
        )
        status_row.addWidget(self.llm_label)

        main_layout.addLayout(status_row)

        # -- Bottom row: Three separate metric panels --
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(10)

        def create_metric(title, initial_text):
            panel = QFrame()
            panel.setStyleSheet("""
                QFrame {
                    background-color: #242424;
                    border-radius: 6px;
                }
            """)

            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(8, 8, 8, 8)
            panel_layout.setSpacing(5)

            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("""
                color: #bbbbbb;
                font-size: 11px;
                font-weight: bold;
                background: transparent;
            """)

            stats_label = QLabel(initial_text)
            stats_label.setAlignment(Qt.AlignCenter)
            stats_label.setWordWrap(True)
            stats_label.setStyleSheet("""
                color: #888888;
                font-size: 11px;
                background: transparent;
            """)

            panel_layout.addWidget(title_label)
            panel_layout.addWidget(stats_label)

            panel.setMinimumWidth(150)
            panel.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred
            )

            return panel, stats_label

        # Tokens per second
        speed_panel, self.llm_speed_stats_label = create_metric(
            "Generation Speed",
            "Avg: 0.00 t/s\nMin: 0.00 t/s\nMax: 0.00 t/s"
        )

        # Response latency
        latency_panel, self.llm_stats_label = create_metric(
            "Response Latency",
            "Avg: 0.00s\nMin: 0.00s\nMax: 0.00s"
        )

        # Generated tokens
        tokens_panel, self.llm_token_stats_label = create_metric(
            "Generated Tokens",
            "Avg: 0 tokens\nMin: 0 tokens\nMax: 0 tokens"
        )

        metrics_layout.addWidget(speed_panel, stretch=1)
        metrics_layout.addWidget(latency_panel, stretch=1)
        metrics_layout.addWidget(tokens_panel, stretch=1)

        main_layout.addLayout(metrics_layout)

        return bar


    def _make_conversation_view(self):
        self.conv_view = QTextEdit()
        self.conv_view.setReadOnly(True)
        self.conv_view.setFont(QFont("Segoe UI", 11))
        # Soft background so it's visually distinct from the window
        self.conv_view.setStyleSheet(
            "QTextEdit { background: #1a1a1a; border-radius: 8px; padding: 8px; }"
        )
        return self.conv_view

    # ── Signal wiring ────────────────────────────────────────────────

    def _connect_signals(self):
        self._bridge.mic_volume.connect(self._on_mic_volume)
        self._bridge.asr_recording.connect(self._on_asr_recording)
        self._bridge.asr_transcribing.connect(self._on_asr_transcribing)
        self._bridge.llm_state.connect(self._on_llm_state)
        self._bridge.conversation.connect(self._on_conversation)
        self._bridge.asr_stats_updated.connect(self._on_asr_stats)
        self._bridge.transcript.connect(self._on_transcript)
        self._bridge.tts_stats_updated.connect(self._on_tts_stats)
        self._bridge.llm_stats_updated.connect(self._on_llm_stats)
        self._bridge.llm_token_updated.connect(self._on_llm_token_stats)
        self._bridge.llm_speed_updated.connect(self._on_llm_speed_stats)

    # ── Slots ────────────────────────────────────────────────────────

    @Slot(float)
    def _on_mic_volume(self, rms: float):
        # rms is typically 0–0.2 for speech; scale to 0–100
        self.vol_bar.setValue(min(100, int(rms * 500)))

    @Slot(float, float, float)
    def _on_asr_stats(self, avg_t, min_t, max_t,):
        self.asr_stats_label.setText(
            f"avg: {avg_t:.2f}s | min: {min_t:.2f}s | max: {max_t:.2f}s"
        )

    @Slot(float, float, float)
    def _on_tts_stats(self, avg_t, min_t, max_t,):
        self.tts_stats_label.setText(
            f"avg: {avg_t:.2f}s | min: {min_t:.2f}s | max: {max_t:.2f}s"
        )

    @Slot(float, float, float)
    def _on_llm_stats(self, avg_t, min_t, max_t,):
        self.llm_stats_label.setText(
            f"avg: {avg_t:.2f}s | min: {min_t:.2f}s | max: {max_t:.2f}s"
        )

    @Slot(float, float, float)
    def _on_llm_token_stats(self, avg_c, min_c, max_c,):
        self.llm_token_stats_label.setText(
            f"avg: {avg_c:.0f} tokens | min: {min_c:.0f} tokens | max: {max_c:.0f} tokens"
        )

    @Slot(float, float, float)
    def _on_llm_speed_stats(self, avg_s, min_s, max_s,):
        self.llm_speed_stats_label.setText(
            f"avg: {avg_s:.2f} t/s | min: {min_s:.2f} t/s | max: {max_s:.2f} t/s"
        )

    @Slot(str)
    def _on_transcript(self, text: str):
        self.script_view.setPlainText(text)

    @Slot(bool)
    def _on_asr_recording(self, recording: bool):
        if recording:
            self.mic_label.setText("● Recording")
            self.mic_label.setStyleSheet("color: #ef5350; font-weight: 500;")
        else:
            self.mic_label.setText("● Idle")
            self.mic_label.setStyleSheet("color: gray; font-weight: 500;")

    @Slot(bool)
    def _on_asr_transcribing(self, transcribing: bool):
        if transcribing:
            self.transcribe_label.setText("● Transcribing…")
            self.transcribe_label.setStyleSheet("color: #42a5f5; font-weight: 500;")
        else:
            self.transcribe_label.setText("○ Not transcribing")
            self.transcribe_label.setStyleSheet("color: gray; font-weight: 500;")

    @Slot(bool)
    def _on_llm_state(self, thinking: bool):
        if thinking:
            self.llm_label.setText("◎ Thinking…")
            self.llm_label.setStyleSheet("color: #ffa726; font-weight: 500;")
        else:
            self.llm_label.setText("○ Ready")
            self.llm_label.setStyleSheet("color: gray; font-weight: 500;")

    @Slot(str, str, str, list)
    def _on_conversation(self, role: str, content: str, reasoning_content: str | None, tool_calls: list | None):
        cursor = self.conv_view.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Role header
        header_fmt = QTextCharFormat()
        header_fmt.setFontWeight(QFont.Bold)
        header_fmt.setForeground(
            QColor("#64b5f6") if role == "user" else QColor("#ffb74d") if role == "tool" else QColor("#a5d6a7")
        )
        if role == "user":
            label = "You"
        elif role == "tool":
            label = "Tool"
        else:
            label = "Shira"
        cursor.insertText(f"{label}\n", header_fmt)

        if reasoning_content:
            # Reasoning content (if present)
            reasoning_fmt = QTextCharFormat()
            reasoning_fmt.setForeground(QColor("#999999"))
            reasoning_fmt.setFontItalic(True)
            cursor.insertText(f"{reasoning_content}\n", reasoning_fmt)

        if tool_calls:
            # Tool calls (if present)
            tool_fmt = QTextCharFormat()
            tool_fmt.setForeground(QColor("#e7ca9d"))
            for call in tool_calls:
                raw_args = call["function"].get("arguments", "{}")
                cursor.insertText(f"[Tool] {call['function']['name']}: {raw_args}\n", tool_fmt)


        # Message body
        body_fmt = QTextCharFormat()
        body_fmt.setForeground(QColor("#dddddd"))

        if role == "tool":
            body_fmt.setForeground(QColor("#e7ca9d"))

        cursor.insertText(f"{content}\n\n", body_fmt)

        self.conv_view.setTextCursor(cursor)
        self.conv_view.ensureCursorVisible()
