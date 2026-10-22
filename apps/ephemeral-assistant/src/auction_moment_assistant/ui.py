from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime
import json
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import traceback

from .map_editor import MapEditor
from .models import CatalogItem
from .observations import ObservationStore
from .presentation import format_prediction
from .predictor import PredictionResult
from .runtime import (
    CapturePipeline,
    EphemeralStateMonitor,
    InferenceCoordinator,
)
from .semantics import event_semantics


MIN_MAIN_PANE_READY_WIDTH = 900
MIN_MAP_PANE_WIDTH = 560
MIN_REVIEW_PANE_WIDTH = 420
INITIAL_SASH_RETRY_MS = 50
INITIAL_SASH_MAX_ATTEMPTS = 12


def preferred_main_sash(width: int) -> int | None:
    """Return a safe initial split only after the paned window has real geometry."""
    width = int(width)
    if width < MIN_MAIN_PANE_READY_WIDTH:
        return None
    return max(
        MIN_MAP_PANE_WIDTH,
        min(int(width * 0.62), width - MIN_REVIEW_PANE_WIDTH),
    )


class AssistantWindow:
    def __init__(
        self,
        *,
        store: ObservationStore,
        pipeline: CapturePipeline | None,
        inference: InferenceCoordinator,
        catalog: tuple[CatalogItem, ...],
        monitor=None,
        title: str = "Auction Moment · 半自动 OCR 预测助手",
    ) -> None:
        self.store = store
        self.pipeline = pipeline
        self.inference = inference
        self.catalog = catalog
        # Packet capture brings its own monitor; otherwise watch the screen.
        self.monitor = monitor if monitor is not None else EphemeralStateMonitor(pipeline, store)
        self._prediction_future: Future | None = None
        self._log_entry_count = 0
        self._initial_sash_attempts = 0
        self._initial_sash_applied = False

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1440x940")
        self.root.minsize(1180, 780)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self.root.after_idle(self._set_initial_sashes)
        self.refresh()
        self.monitor.start()
        self.root.after(100, self._poll_monitor)

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="正在连接并监视窗口…")
        ttk.Label(
            top,
            textvariable=self.status_var,
            font=("Microsoft YaHei UI", 12, "bold"),
            foreground="#0f172a",
        ).pack(side=tk.LEFT)
        self.round_var = tk.IntVar(value=1)
        ttk.Label(top, text="轮次").pack(side=tk.LEFT, padx=(22, 4))
        ttk.Spinbox(top, from_=1, to=5, width=4, textvariable=self.round_var).pack(
            side=tk.LEFT
        )
        ttk.Button(top, text="修正轮次", command=self.apply_round).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(top, text="立即重扫本轮", command=self.request_scan).pack(
            side=tk.LEFT, padx=(12, 4)
        )
        self.pause_button = ttk.Button(
            top, text="暂停监视", command=self.toggle_pause
        )
        self.pause_button.pack(side=tk.LEFT)
        ttk.Button(top, text="清空本局内存", command=self.reset).pack(side=tk.RIGHT)

        ttk.Label(
            self.root,
            text=(
                "只读取画面；自动输入仅限左侧地图滚动。无抓包、无点击、无按键、"
                "无自动出价、无截图/事件/预测/结算落盘。"
            ),
            padding=(8, 0),
            foreground="#64748b",
        ).pack(fill=tk.X)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane = body
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.LabelFrame(body, text="地图建模与快速纠错", padding=8)
        right = ttk.Frame(body)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=2)
        right.rowconfigure(1, weight=5)
        body.add(left, weight=3)
        body.add(right, weight=2)
        self.map_editor = MapEditor(
            left, store=self.store, catalog=self.catalog, on_change=self._changed
        )
        self.map_editor.pack(fill=tk.BOTH, expand=True)

        event_box = ttk.LabelFrame(right, text="事件列表与人工修正", padding=8)
        event_box.grid(row=0, column=0, sticky=tk.NSEW)
        columns = ("round", "kind", "text", "confidence", "source")
        self.event_tree = ttk.Treeview(
            event_box, columns=columns, show="headings", height=7
        )
        for name, label, width in (
            ("round", "轮", 36),
            ("kind", "类型", 55),
            ("text", "识别文本", 240),
            ("confidence", "置信", 52),
            ("source", "来源", 48),
        ):
            self.event_tree.heading(name, text=label)
            self.event_tree.column(name, width=width, anchor=tk.W)
        self.event_tree.pack(fill=tk.BOTH, expand=True)
        self.event_tree.bind("<<TreeviewSelect>>", self._select_event)

        editor = ttk.Frame(event_box)
        editor.pack(fill=tk.X, pady=(7, 0))
        self.event_label_var = tk.StringVar(value="选择上方事件")
        ttk.Label(editor, textvariable=self.event_label_var, width=14).pack(
            side=tk.LEFT
        )
        self.event_text_var = tk.StringVar()
        ttk.Entry(editor, textvariable=self.event_text_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        ttk.Button(editor, text="应用修正", command=self.apply_event).pack(
            side=tk.RIGHT
        )

        lower = ttk.Frame(right)
        lower.grid(row=1, column=0, sticky=tk.NSEW, pady=(8, 0))

        evidence_box = ttk.LabelFrame(lower, text="本轮证据", padding=8)
        evidence_box.pack(fill=tk.X, pady=(8, 0))
        self.bankroll_var = tk.StringVar()
        self.map_rows_var = tk.StringVar()
        self.map_exact_var = tk.BooleanVar(value=False)
        self.proof_var = tk.StringVar(value="等待自动扫描")
        ttk.Label(evidence_box, text="持有金额").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(evidence_box, textvariable=self.bankroll_var, width=16).grid(
            row=0, column=1, sticky=tk.EW, padx=(4, 10)
        )
        ttk.Button(evidence_box, text="修正金额", command=self.apply_bankroll).grid(
            row=0, column=2
        )
        ttk.Label(evidence_box, text="地图行数").grid(
            row=1, column=0, sticky=tk.W, pady=(6, 0)
        )
        ttk.Entry(evidence_box, textvariable=self.map_rows_var, width=8).grid(
            row=1, column=1, sticky=tk.W, padx=(4, 10), pady=(6, 0)
        )
        ttk.Checkbutton(
            evidence_box,
            text="已人工确认到底部",
            variable=self.map_exact_var,
        ).grid(row=1, column=1, sticky=tk.E, pady=(6, 0))
        ttk.Button(evidence_box, text="应用行数", command=self.apply_map_extent).grid(
            row=1, column=2, pady=(6, 0)
        )
        ttk.Label(
            evidence_box,
            textvariable=self.proof_var,
            foreground="#475569",
            wraplength=420,
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(7, 0))
        evidence_box.columnconfigure(1, weight=1)

        prediction_box = ttk.LabelFrame(
            lower,
            text="v6 + world-model-v2 估值与建议出价（仅人工参考）",
            padding=8,
        )
        prediction_box.pack(fill=tk.X, pady=(8, 0))
        self.prediction_var = tk.StringVar(value="等待每轮自动扫描")
        ttk.Label(
            prediction_box,
            textvariable=self.prediction_var,
            justify=tk.LEFT,
            wraplength=470,
        ).pack(fill=tk.X)

        log_box = ttk.LabelFrame(
            lower,
            text="运行与诊断日志（仅内存，不写文件）",
            padding=8,
        )
        log_box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        log_actions = ttk.Frame(log_box)
        log_actions.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(
            log_actions,
            text="可滚动、选择并 Ctrl+C；异常会保留完整堆栈",
            foreground="#64748b",
        ).pack(side=tk.LEFT)
        ttk.Button(
            log_actions, text="复制全部", command=self.copy_all_logs
        ).pack(side=tk.RIGHT)
        ttk.Button(
            log_actions, text="清空显示", command=self.clear_logs
        ).pack(side=tk.RIGHT, padx=(0, 5))
        self.log_text = scrolledtext.ScrolledText(
            log_box,
            height=14,
            wrap=tk.WORD,
            font=("Cascadia Mono", 9),
            undo=False,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)
        self.log_menu = tk.Menu(self.root, tearoff=False)
        self.log_menu.add_command(label="复制选中", command=self.copy_selected_log)
        self.log_menu.add_command(label="复制全部", command=self.copy_all_logs)
        self.log_menu.add_separator()
        self.log_menu.add_command(label="全选", command=self.select_all_logs)
        self.log_text.bind("<Button-3>", self._show_log_menu)
        self._append_log("startup", "界面已启动；日志只保留在本进程内存中")

    def _set_initial_sashes(self) -> None:
        if self._initial_sash_applied:
            return
        self.root.update_idletasks()
        target = preferred_main_sash(self.main_pane.winfo_width())
        if target is None:
            self._retry_initial_sash()
            return
        try:
            self.main_pane.sashpos(0, target)
            self.root.update_idletasks()
            actual = self.main_pane.sashpos(0)
        except tk.TclError:
            return
        if actual < MIN_MAP_PANE_WIDTH // 2:
            self._retry_initial_sash()
            return
        self._initial_sash_applied = True

    def _retry_initial_sash(self) -> None:
        self._initial_sash_attempts += 1
        if self._initial_sash_attempts > INITIAL_SASH_MAX_ATTEMPTS:
            return
        try:
            self.root.after(INITIAL_SASH_RETRY_MS, self._set_initial_sashes)
        except tk.TclError:
            return

    def _append_log(
        self,
        category: str,
        message: str,
        details: tuple[str, ...] | list[str] = (),
    ) -> None:
        self._log_entry_count += 1
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        lines = [
            f"[{stamp}] #{self._log_entry_count:04d} [{category}] {message}"
        ]
        for detail in details:
            cleaned = str(detail).rstrip()
            if cleaned:
                lines.extend(f"    {line}" for line in cleaned.splitlines())
        block = "\n".join(lines) + "\n"
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, block)
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    @staticmethod
    def _scan_log_details(scan) -> tuple[str, ...]:
        details = [
            (
                f"revision={scan.revision} success={scan.success} "
                f"round={scan.round_number} viewports={scan.viewport_count} "
                f"map_items={scan.map_item_count} "
                f"estimated_rows={scan.estimated_rows} "
                f"alignment_stable={scan.alignment_stable}"
            )
        ]
        if scan.ocr is not None:
            details.append(
                (
                    f"ocr.round={scan.ocr.round_number} "
                    f"round_verified={scan.ocr.round_verified} "
                    f"bankroll={scan.ocr.bankroll} "
                    f"timing_ms={scan.ocr.timing_ms:.2f}"
                )
            )
            for event in scan.ocr.events:
                details.append(
                    (
                        f"ocr.event R{event.round_number} {event.kind} "
                        f"confidence={event.confidence:.4f} "
                        f"source={event.source} text={event.text!r} "
                        "semantics="
                        + json.dumps(
                            event.semantics,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                )
        if scan.errors:
            details.append("scan.errors=" + " | ".join(scan.errors))
        return tuple(details)

    def _show_log_menu(self, event) -> None:
        try:
            self.log_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.log_menu.grab_release()

    def copy_selected_log(self) -> None:
        try:
            content = self.log_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            self.status_var.set("日志区尚未选择文字")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_var.set("已复制选中的日志")

    def copy_all_logs(self) -> None:
        content = self.log_text.get("1.0", "end-1c")
        if not content:
            self.status_var.set("当前没有可复制的日志")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_var.set("已复制全部内存日志")

    def select_all_logs(self) -> None:
        self.log_text.tag_add(tk.SEL, "1.0", "end-1c")
        self.log_text.mark_set(tk.INSERT, "1.0")
        self.log_text.see("1.0")
        self.log_text.focus_set()

    def clear_logs(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self._log_entry_count = 0
        self._append_log("log", "日志显示已由用户手动清空")

    def _changed(self) -> None:
        snapshot = self.store.snapshot()
        self._append_log(
            "evidence",
            (
                f"证据已更新 revision={snapshot.revision} "
                f"round={snapshot.round_number} events={len(snapshot.events)} "
                f"map_items={len(snapshot.map_items)}"
            ),
        )
        self.refresh()
        self.request_prediction()

    def _poll_monitor(self) -> None:
        for update in self.monitor.drain():
            details = list(update.details)
            if update.scan is not None:
                details = [*self._scan_log_details(update.scan), *details]
                snapshot = self.store.snapshot()
                details.extend(
                    (
                        "snapshot.events="
                        + json.dumps(
                            snapshot.events,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "snapshot.map_items="
                        + json.dumps(
                            snapshot.map_items,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                )
            self._append_log(update.kind, update.message, details)
            self.round_var.set(update.round_number)
            self.status_var.set(update.message)
            if update.kind in {
                "session_started",
                "session_ended",
                "round_changed",
                "scan_finished",
            }:
                self.refresh()
            if update.kind == "scan_finished":
                if update.scan and update.scan.errors:
                    self.status_var.set(
                        update.message + "；" + "；".join(update.scan.errors)
                    )
                self.request_prediction()
        self.root.after(100, self._poll_monitor)

    def toggle_pause(self) -> None:
        paused = not self.monitor.paused
        self.monitor.set_paused(paused)
        self.pause_button.configure(text="继续监视" if paused else "暂停监视")
        self.status_var.set("监视已暂停" if paused else "监视已恢复")
        self._append_log("control", "监视已暂停" if paused else "监视已恢复")

    def request_scan(self) -> None:
        self.monitor.request_scan(self.round_var.get())
        self.status_var.set("已请求内存重扫；仅会滚动左侧地图")
        self._append_log("control", f"请求手动重扫 R{self.round_var.get()}")

    def apply_round(self) -> None:
        self.monitor.set_round(self.round_var.get())
        self.status_var.set(f"已人工修正为第 {self.round_var.get()} 轮")
        self._append_log("human", f"人工修正轮次为 R{self.round_var.get()}")
        self.refresh()

    def _select_event(self, _event=None) -> None:
        selection = self.event_tree.selection()
        if not selection:
            return
        iid = selection[0]
        round_text, kind = iid.split(":", 1)
        values = self.event_tree.item(iid, "values")
        self.event_label_var.set(
            f"R{round_text} {'公共' if kind == 'public' else '个人'}"
        )
        self.event_text_var.set(str(values[2] or ""))

    def apply_event(self) -> None:
        selection = self.event_tree.selection()
        if not selection:
            messagebox.showinfo("尚未选择", "先在事件列表中选择一条")
            return
        round_text, kind = selection[0].split(":", 1)
        text = self.event_text_var.get().strip()
        self.store.correct_event(
            int(round_text), kind, text, event_semantics(text)
        )
        self._append_log(
            "human",
            f"人工修正 R{round_text} {kind}事件",
            (f"text={text!r}",),
        )
        self._changed()

    def apply_bankroll(self) -> None:
        raw = self.bankroll_var.get().replace(",", "").strip()
        try:
            value = int(raw) if raw else None
        except ValueError:
            messagebox.showerror("金额无效", "持有金额必须是整数")
            return
        self.store.correct_bankroll(value)
        self._append_log("human", f"人工修正持有金额为 {value}")
        self._changed()

    def apply_map_extent(self) -> None:
        raw = self.map_rows_var.get().strip()
        try:
            rows = int(raw) if raw else None
            self.store.correct_map_extent(rows, bool(self.map_exact_var.get()))
        except ValueError as exc:
            messagebox.showerror("地图行数无效", str(exc))
            return
        self._append_log(
            "human",
            (
                f"人工修正地图行数为 {rows}；"
                f"到底部确认={bool(self.map_exact_var.get())}"
            ),
        )
        self._changed()

    def request_prediction(self) -> None:
        self._prediction_future = self.inference.submit(self.store)
        self._append_log(
            "prediction", f"提交 revision={self.store.revision} 的估值任务"
        )
        self.root.after(35, self._poll_prediction)

    def _poll_prediction(self) -> None:
        future = self._prediction_future
        if future is None or not future.done():
            self.root.after(35, self._poll_prediction)
            return
        try:
            result: PredictionResult = future.result()
        except Exception as exc:
            self.prediction_var.set(f"预测失败：{exc}")
            self._append_log(
                "prediction-error",
                f"预测失败：{type(exc).__name__}: {exc}",
                (traceback.format_exc(),),
            )
            return
        if not self.inference.current(self.store, result):
            self._append_log(
                "prediction-stale",
                (
                    f"丢弃旧结果 revision={result.revision}；"
                    f"当前 revision={self.store.revision}"
                ),
            )
            self.request_prediction()
            return
        self.prediction_var.set(format_prediction(result))
        self._append_log(
            "prediction-result",
            (
                f"revision={result.revision} status={result.status} "
                f"compatible_worlds={result.compatible_worlds} "
                f"source={result.estimate_source}"
            ),
            (
                f"contract={result.contract}",
                f"p10={result.p10} p50={result.p50} p90={result.p90}",
                "issues=" + json.dumps(result.issues, ensure_ascii=False),
                "diagnostics="
                + json.dumps(result.diagnostics, ensure_ascii=False),
            ),
        )
        if self.inference.refine(self.store, result):
            self._append_log(
                "prediction-refine",
                f"revision={result.revision} 预算内未收敛，后台继续细化",
            )
            self._refining_revision = result.revision
            self.root.after(250, lambda: self._poll_refinement(result.revision))

    def _poll_refinement(self, revision: int) -> None:
        # A newer refinement has its own poll loop.
        if getattr(self, "_refining_revision", None) != revision:
            return
        state, result = self.inference.refined(self.store)
        if state == "running":
            self.root.after(250, lambda: self._poll_refinement(revision))
            return
        self._refining_revision = None
        if result is None:
            self._append_log(
                "prediction-refine",
                "后台细化结束，保留原估值"
                if state == "done"
                else "后台细化已取消或被新任务替代",
            )
            return
        self.prediction_var.set(format_prediction(result))
        seconds = (result.conditional_diagnostics or {}).get("exact_refine_seconds")
        self._append_log(
            "prediction-refined",
            (
                f"revision={result.revision} status={result.status} "
                f"细化 {seconds} 秒后收敛"
            ),
            (f"p10={result.p10} p50={result.p50} p90={result.p90}",),
        )

    def refresh(self) -> None:
        snapshot = self.store.snapshot()
        self.round_var.set(snapshot.round_number)
        self.bankroll_var.set(
            f"{snapshot.bankroll:,}" if snapshot.bankroll is not None else ""
        )
        self.map_rows_var.set(
            str(snapshot.map_rows) if snapshot.map_rows is not None else ""
        )
        self.map_exact_var.set(snapshot.map_height_exact)
        self.proof_var.set(
            "扫描完整性："
            f"{snapshot.completeness_source}；地图行数：{snapshot.map_rows_source}；"
            f"出价前证明：{snapshot.pre_bid_source}"
        )
        selected = self.event_tree.selection()
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)
        events = {
            (int(value["round_number"]), str(value["kind"])): value
            for value in snapshot.events
        }
        for round_number in range(1, snapshot.round_number + 1):
            for kind in ("public", "personal"):
                value = events.get((round_number, kind), {})
                iid = f"{round_number}:{kind}"
                self.event_tree.insert(
                    "",
                    tk.END,
                    iid=iid,
                    values=(
                        round_number,
                        "公共" if kind == "public" else "个人",
                        value.get("text") or "（待识别/可手填）",
                        f"{float(value.get('confidence') or 0):.2f}",
                        "人工" if value.get("human_locked") else "OCR",
                    ),
                )
        if selected and self.event_tree.exists(selected[0]):
            self.event_tree.selection_set(selected[0])
        self.map_editor.refresh()

    def reset(self) -> None:
        self.store.reset()
        self.monitor.set_round(1)
        self.prediction_var.set("等待每轮自动扫描")
        self.status_var.set("本局临时状态已从内存清空，继续监视大厅")
        self._append_log(
            "human", "清空本局临时证据；运行日志按要求继续保留"
        )
        self.refresh()

    def close(self) -> None:
        self.monitor.close()
        self.store.reset()
        self.inference.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
