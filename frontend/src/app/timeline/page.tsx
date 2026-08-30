"use client";

import {
  CalendarDays,
  ChevronDown,
  ExternalLink,
  Eye,
  GitBranch,
  RotateCcw,
  SlidersHorizontal,
  X,
} from "lucide-react";
import Image, { type ImageProps } from "next/image";
import { useEffect, useMemo, useState } from "react";
import { apiGet } from "@/lib/api/client";
import type { ProjectTimelineItem, ProjectsResponse } from "@/lib/api/types";
import { AutoReviewBadge } from "@/components/review/AutoReviewBadge";
import todoIcon from "@/app/timeline/icons/todo.png";
import slackIcon from "@/app/timeline/icons/slack.svg";
import gmailIcon from "@/app/timeline/icons/gmail.png";
import driveIcon from "@/app/timeline/icons/drive.svg";
import calendarIcon from "@/app/timeline/icons/calendar.svg";

type TimelineSource = "Slack" | "Gmail" | "Drive" | "Calendar" | "Source";
type FilterableTimelineSource = Exclude<TimelineSource, "Source">;
type PeriodFilter = "all" | "7d" | "30d";
type SourceFilter = "all" | FilterableTimelineSource;
type TimelineStatus = "승인" | "완료";
type StatusFilter = "all" | TimelineStatus;

type TimelineHistory = {
  id: string;
  itemType: string;
  createdAt: string;
  time: string;
  source: TimelineSource;
  title: string;
  summary: string;
  history: string;
  status: TimelineStatus;
  resolutionSource?: "human" | "auto_policy" | null;
  sourceUrl: string;
  preview: string;
  snippets: { author: string; body: string; time: string }[];
};

type ProjectTimeline = {
  id: string;
  name: string;
  histories: TimelineHistory[];
};

type TimelineDateGroupData = {
  dateKey: string;
  dateLabel: string;
  monthKey: string;
  monthLabel: string;
  items: TimelineHistory[];
};

type TimelineMonthGroupData = {
  monthKey: string;
  monthLabel: string;
  dateGroups: TimelineDateGroupData[];
};

export default function TimelinePage() {
  const [projectTimelines, setProjectTimelines] = useState<ProjectTimeline[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState("");
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | undefined>();
  const [expandedMonthKeys, setExpandedMonthKeys] = useState<Set<string>>(() => new Set());
  const [expandedDateKeys, setExpandedDateKeys] = useState<Set<string>>(() => new Set());
  const [expandedItemGroups, setExpandedItemGroups] = useState<Set<string>>(() => new Set());
  const [showAllDates, setShowAllDates] = useState(false);
  const [periodFilter, setPeriodFilter] = useState<PeriodFilter>("all");
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("all");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();

  useEffect(() => {
    let active = true;

    apiGet<ProjectsResponse>("/api/v1/projects")
      .then((response) => {
        if (!active) return;
        const projects = response.projects.map((project) => ({
          id: project.project_key,
          name: project.name,
          histories: project.timeline_items
            .filter((item) => item.review_status === "approved")
            .map(timelineHistoryFromProjectItem),
        }));
        const defaultProject = firstProjectWithHistories(projects);
        const defaultGroups = groupHistoriesByDate(defaultProject?.histories ?? []);
        const defaultMonthGroups = groupDateGroupsByMonth(defaultGroups);
        setProjectTimelines(projects);
        setSelectedProjectId((current) => current || defaultProject?.id || "");
        setExpandedMonthKeys((current) => current.size > 0 ? current : defaultExpandedMonthKeys(defaultMonthGroups));
        setExpandedDateKeys((current) => current.size > 0 ? current : defaultExpandedDateKeys(defaultGroups));
      })
      .catch((caught) => {
        if (active) setError(caught instanceof Error ? caught.message : "Failed to load timeline.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  const selectedProject = useMemo(
    () => projectTimelines.find((project) => project.id === selectedProjectId) ?? projectTimelines[0],
    [projectTimelines, selectedProjectId],
  );
  const filteredHistories = useMemo(
    () => filterHistories(selectedProject?.histories ?? [], periodFilter, sourceFilter, statusFilter),
    [periodFilter, selectedProject?.histories, sourceFilter, statusFilter],
  );
  const selectedHistory = selectedProject?.histories.find((history) => history.id === selectedHistoryId);
  const historyCount = selectedProject?.histories.length ?? 0;
  const groupedHistories = useMemo(
    () => groupHistoriesByDate(filteredHistories, showAllDates),
    [filteredHistories, showAllDates],
  );
  const groupedHistoriesByMonth = useMemo(() => groupDateGroupsByMonth(groupedHistories), [groupedHistories]);

  useEffect(() => {
    setExpandedMonthKeys(defaultExpandedMonthKeys(groupedHistoriesByMonth));
    setExpandedDateKeys(defaultExpandedDateKeys(groupedHistories));
    setExpandedItemGroups(new Set());
  }, [groupedHistories, groupedHistoriesByMonth]);

  if (loading || !selectedProject) {
    return (
      <div className="reference-dashboard space-y-4">
        <section className="page-heading reference-heading">
          <div>
            <p className="text-[13px] font-bold text-[var(--primary-dark)]">Timeline</p>
            <h1>타임라인</h1>
            <p>{loading ? "승인된 프로젝트 워크플로우를 불러오고 있습니다." : error || "승인된 프로젝트 타임라인 항목이 아직 없습니다."}</p>
          </div>
          <div className="panel inline-flex h-fit w-fit items-center gap-2 px-4 py-3 text-[13px] font-bold">
            <GitBranch className="h-4 w-4 text-[var(--primary)]" aria-hidden="true" />
            0 histories
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="reference-dashboard space-y-4">
      <section className="flex flex-col gap-4 rounded-[2rem] border border-white/80 bg-white/75 p-5 shadow-[0_24px_70px_rgba(15,23,42,0.06)] backdrop-blur sm:p-6 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-sm font-bold text-indigo-500">Timeline</p>
          <h1 className="mt-1 text-[2.25rem] font-extrabold tracking-tight text-slate-950">타임라인</h1>
          <p className="mt-3 max-w-3xl text-[15px] font-medium leading-7 text-slate-500">
            등록된 프로젝트에 연결된 승인 타임라인 항목을 시간순으로 확인합니다.
          </p>
        </div>
        <div className="mt-4 inline-flex h-fit w-fit items-center gap-2 rounded-2xl border border-slate-200/80 bg-white px-4 py-3 text-[13px] font-bold text-slate-700 shadow-sm lg:mt-0">
          <GitBranch className="h-4 w-4 text-indigo-600" aria-hidden="true" />
          {historyCount.toLocaleString()} histories
        </div>
      </section>

      <TimelineStatStrip histories={selectedProject.histories} />

      <nav className="rounded-[2rem] border border-white/80 bg-white/75 p-2 shadow-[0_20px_60px_rgba(15,23,42,0.06)] backdrop-blur" aria-label="Timeline selector">
        <div className="flex gap-2 overflow-x-auto">
        {projectTimelines.map((project) => (
          <button
            key={project.id}
            type="button"
            aria-pressed={project.id === selectedProjectId}
            className={`shrink-0 rounded-full px-4 py-2 text-left text-[13px] font-extrabold transition ${
              project.id === selectedProjectId
                ? "bg-indigo-600 text-white shadow-[0_14px_30px_rgba(79,70,229,0.22)]"
                : "bg-white text-slate-600 shadow-sm hover:bg-indigo-50 hover:text-indigo-700"
            }`}
            onClick={() => {
              setSelectedProjectId(project.id);
              setSelectedHistoryId(undefined);
              const projectDateGroups = groupHistoriesByDate(project.histories, showAllDates);
              const projectMonthGroups = groupDateGroupsByMonth(projectDateGroups);
              setExpandedMonthKeys(defaultExpandedMonthKeys(projectMonthGroups));
              setExpandedDateKeys(defaultExpandedDateKeys(projectDateGroups));
              setExpandedItemGroups(new Set());
            }}
          >
            {project.name}
            <span className="ml-2 rounded-full bg-white/25 px-2 py-0.5 text-[11px]">{project.histories.length.toLocaleString()}</span>
          </button>
        ))}
        </div>
      </nav>

      <section className={`timeline-history-layout ${selectedHistory ? "history-open" : ""}`}>
        <TimelineListPanel
          histories={filteredHistories}
          groupedHistoriesByMonth={groupedHistoriesByMonth}
          expandedMonthKeys={expandedMonthKeys}
          expandedDateKeys={expandedDateKeys}
          expandedItemGroups={expandedItemGroups}
          selectedHistoryId={selectedHistoryId}
          showAllDates={showAllDates}
          periodFilter={periodFilter}
          sourceFilter={sourceFilter}
          statusFilter={statusFilter}
          onToggleShowAllDates={() => setShowAllDates((current) => !current)}
          onChangePeriod={setPeriodFilter}
          onChangeSource={setSourceFilter}
          onChangeStatus={setStatusFilter}
          onResetFilters={() => {
            setPeriodFilter("all");
            setSourceFilter("all");
            setStatusFilter("all");
          }}
          onToggleMonth={(monthKey) => {
            setExpandedMonthKeys((current) => {
              const next = new Set(current);
              if (next.has(monthKey)) {
                next.delete(monthKey);
              } else {
                next.add(monthKey);
              }
              return next;
            });
          }}
          onToggleDate={(dateKey) => {
            setExpandedDateKeys((current) => {
              const next = new Set(current);
              if (next.has(dateKey)) {
                next.delete(dateKey);
              } else {
                next.add(dateKey);
              }
              return next;
            });
            setExpandedItemGroups((current) => {
              const next = new Set(current);
              next.delete(dateKey);
              return next;
            });
            setSelectedHistoryId(undefined);
          }}
          onToggleMore={(dateKey) => {
            setExpandedItemGroups((current) => {
              const next = new Set(current);
              if (next.has(dateKey)) {
                next.delete(dateKey);
              } else {
                next.add(dateKey);
              }
              return next;
            });
          }}
          onJumpDate={(dateKey) => {
            setExpandedMonthKeys((current) => new Set(current).add(dateKey.slice(0, 7)));
            setExpandedDateKeys((current) => new Set(current).add(dateKey));
            setSelectedHistoryId(undefined);
            document.getElementById(`timeline-date-${dateKey}`)?.scrollIntoView({ block: "start", behavior: "smooth" });
          }}
          onSelectHistory={(historyId) => {
            setSelectedHistoryId((current) => (current === historyId ? undefined : historyId));
          }}
        />

        {selectedHistory ? (
          <aside className="timeline-history-panel h-fit rounded-[2rem] border border-white/80 bg-white/85 p-5 shadow-[0_24px_70px_rgba(15,23,42,0.08)] backdrop-blur">
            <div className="flex items-start justify-between gap-3 border-b border-line pb-4">
              <div>
                <p className="text-[12px] font-bold text-indigo-600">{selectedHistory.source} history</p>
                <h2 className="mt-1 text-[16px] font-extrabold text-slate-950">{selectedHistory.title}</h2>
              </div>
              <button type="button" className="icon-button small" aria-label="Close history" onClick={() => setSelectedHistoryId(undefined)}>
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>

            <p className="mt-4 text-[13px] leading-6 text-muted">{selectedHistory.history}</p>

            {selectedHistory.snippets.length > 0 ? (
              <div className="mt-4 space-y-3">
                {selectedHistory.snippets.map((snippet, index) => (
                  <div key={`${snippet.time}-${index}`} className="rounded-lg border border-line bg-surface-soft p-3">
                    <div className="flex items-center justify-between gap-2 text-[12px] font-bold text-muted">
                      <span>{snippet.author}</span>
                      <span>{snippet.time}</span>
                    </div>
                    <p className="mt-2 text-[13px] leading-6 text-ink">{snippet.body}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="mt-4 rounded-lg border border-dashed border-line bg-surface-soft p-4 text-[13px] text-muted">
                연결된 source snippet이 없습니다.
              </div>
            )}

            {selectedHistory.sourceUrl ? (
              <a
                href={selectedHistory.sourceUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-4 inline-flex items-center gap-2 text-[12px] font-extrabold text-[var(--primary-dark)] underline-offset-4 hover:underline"
              >
                <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                Open source
              </a>
            ) : null}
          </aside>
        ) : null}
      </section>
    </div>
  );
}

function TimelineListPanel({
  histories,
  groupedHistoriesByMonth,
  expandedMonthKeys,
  expandedDateKeys,
  expandedItemGroups,
  selectedHistoryId,
  showAllDates,
  periodFilter,
  sourceFilter,
  statusFilter,
  onToggleShowAllDates,
  onChangePeriod,
  onChangeSource,
  onChangeStatus,
  onResetFilters,
  onToggleMonth,
  onToggleDate,
  onToggleMore,
  onJumpDate,
  onSelectHistory,
}: {
  histories: TimelineHistory[];
  groupedHistoriesByMonth: TimelineMonthGroupData[];
  expandedMonthKeys: Set<string>;
  expandedDateKeys: Set<string>;
  expandedItemGroups: Set<string>;
  selectedHistoryId?: string;
  showAllDates: boolean;
  periodFilter: PeriodFilter;
  sourceFilter: SourceFilter;
  statusFilter: StatusFilter;
  onToggleShowAllDates: () => void;
  onChangePeriod: (value: PeriodFilter) => void;
  onChangeSource: (value: SourceFilter) => void;
  onChangeStatus: (value: StatusFilter) => void;
  onResetFilters: () => void;
  onToggleMonth: (monthKey: string) => void;
  onToggleDate: (dateKey: string) => void;
  onToggleMore: (dateKey: string) => void;
  onJumpDate: (dateKey: string) => void;
  onSelectHistory: (historyId: string) => void;
}) {
  return (
    <article className="overflow-visible rounded-[2rem] border border-white/80 bg-white/75 p-0 shadow-[0_24px_70px_rgba(15,23,42,0.06)] backdrop-blur">
      <TimelineSummaryBar
        count={histories.length}
        periodFilter={periodFilter}
        sourceFilter={sourceFilter}
        statusFilter={statusFilter}
        showAllDates={showAllDates}
        onToggleShowAllDates={onToggleShowAllDates}
        onChangePeriod={onChangePeriod}
        onChangeSource={onChangeSource}
        onChangeStatus={onChangeStatus}
        onResetFilters={onResetFilters}
      />

      <div className="border-t border-slate-200/70 bg-[#f8faff] px-3 py-4 sm:px-5">
        {histories.length > 0 ? (
          <div className="grid gap-5 xl:grid-cols-[220px_minmax(0,1fr)]">
            <TimelineDateIndex
              monthGroups={groupedHistoriesByMonth}
              expandedMonthKeys={expandedMonthKeys}
              expandedDateKeys={expandedDateKeys}
              onToggleMonth={onToggleMonth}
              onJumpDate={onJumpDate}
            />
            <div className="min-w-0 space-y-5">
              {groupedHistoriesByMonth.map((monthGroup) => (
                <section key={monthGroup.monthKey} className="space-y-4">
                  <button
                    type="button"
                    aria-expanded={expandedMonthKeys.has(monthGroup.monthKey)}
                    data-testid={`timeline-month-header-${monthGroup.monthKey}`}
                    className="sticky top-24 z-10 flex w-full items-center justify-between rounded-2xl border border-white/80 bg-white/95 px-4 py-3 text-left text-sm font-extrabold text-slate-950 shadow-sm backdrop-blur transition hover:border-indigo-100"
                    onClick={() => onToggleMonth(monthGroup.monthKey)}
                  >
                    <span className="flex min-w-0 items-center gap-3">
                      <span className="grid h-9 w-9 shrink-0 place-items-center rounded-2xl bg-indigo-50 text-indigo-600">
                        <CalendarDays className="h-4 w-4" aria-hidden="true" />
                      </span>
                      <span>
                        <span className="block">{monthGroup.monthLabel}</span>
                        <span className="text-xs font-semibold text-slate-500">
                          {monthGroup.dateGroups.length.toLocaleString()}일 · {monthGroupTotalCount(monthGroup).toLocaleString()}건
                        </span>
                      </span>
                    </span>
                    <span className="flex items-center gap-2">
                      <span className="rounded-full bg-indigo-50 px-2.5 py-1 text-xs font-extrabold text-indigo-700">
                        {monthGroupTotalCount(monthGroup).toLocaleString()}건
                      </span>
                      <ChevronDown className={`h-4 w-4 text-slate-400 transition ${expandedMonthKeys.has(monthGroup.monthKey) ? "rotate-180" : ""}`} aria-hidden="true" />
                    </span>
                  </button>
                  {expandedMonthKeys.has(monthGroup.monthKey) ? (
                    monthGroup.dateGroups.map((group) => (
                      <TimelineDateGroup
                        key={group.dateKey}
                        group={group}
                        isExpanded={expandedDateKeys.has(group.dateKey)}
                        isShowingAll={expandedItemGroups.has(group.dateKey)}
                        selectedHistoryId={selectedHistoryId}
                        onToggleDate={onToggleDate}
                        onToggleMore={onToggleMore}
                        onSelectHistory={onSelectHistory}
                      />
                    ))
                  ) : (
                    <MonthCollapsedSummary monthGroup={monthGroup} onExpand={() => onToggleMonth(monthGroup.monthKey)} />
                  )}
                </section>
              ))}
            </div>
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-line bg-surface-soft p-5 text-[13px] text-muted">
            조건에 맞는 승인 타임라인 항목이 없습니다. 필터를 초기화하거나 Review에서 프로젝트 타임라인 후보를 승인해 주세요.
          </div>
        )}
      </div>
    </article>
  );
}

function TimelineStatStrip({ histories }: { histories: TimelineHistory[] }) {
  const approvedCount = histories.filter((history) => history.status === "승인" || history.status === "완료").length;
  const sourceCounts = histories.reduce<Record<string, number>>((acc, history) => {
    acc[history.source] = (acc[history.source] ?? 0) + 1;
    return acc;
  }, {});
  const mainSource = Object.entries(sourceCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "-";
  const latestDate = histories[0]?.createdAt ? shortDateLabel(dateKeyForValue(histories[0].createdAt)) : "-";

  return (
    <section data-testid="timeline-summary-strip" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <TimelineStat label="전체 히스토리" value={`${histories.length.toLocaleString()}건`} />
      <TimelineStat label="승인됨" value={`${approvedCount.toLocaleString()}건`} />
      <TimelineStat label="주요 소스" value={mainSource} />
      <TimelineStat label="최근 날짜" value={latestDate} />
    </section>
  );
}

function TimelineStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[1.5rem] border border-white/80 bg-white/80 px-4 py-3 shadow-[0_16px_42px_rgba(15,23,42,0.05)]">
      <p className="text-xs font-bold text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-extrabold text-slate-950">{value}</p>
    </div>
  );
}

function TimelineSummaryBar({
  count,
  periodFilter,
  sourceFilter,
  statusFilter,
  showAllDates,
  onToggleShowAllDates,
  onChangePeriod,
  onChangeSource,
  onChangeStatus,
  onResetFilters,
}: {
  count: number;
  periodFilter: PeriodFilter;
  sourceFilter: SourceFilter;
  statusFilter: StatusFilter;
  showAllDates: boolean;
  onToggleShowAllDates: () => void;
  onChangePeriod: (value: PeriodFilter) => void;
  onChangeSource: (value: SourceFilter) => void;
  onChangeStatus: (value: StatusFilter) => void;
  onResetFilters: () => void;
}) {
  return (
    <div className="flex flex-col gap-4 bg-white/70 px-4 py-4 sm:px-5 lg:flex-row lg:items-center lg:justify-between">
      <div className="flex items-center gap-3">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-2xl bg-indigo-50 text-indigo-600">
          <CalendarDays className="h-5 w-5" aria-hidden="true" />
        </span>
        <div>
          <p className="text-[12px] font-bold text-slate-500">필터 결과</p>
          <h2 className="text-[18px] font-extrabold leading-6 text-slate-950">{count.toLocaleString()}건</h2>
        </div>
      </div>
      <TimelineFilterControls
        periodFilter={periodFilter}
        sourceFilter={sourceFilter}
        statusFilter={statusFilter}
        showAllDates={showAllDates}
        onToggleShowAllDates={onToggleShowAllDates}
        onChangePeriod={onChangePeriod}
        onChangeSource={onChangeSource}
        onChangeStatus={onChangeStatus}
        onResetFilters={onResetFilters}
      />
    </div>
  );
}

function TimelineFilterControls({
  periodFilter,
  sourceFilter,
  statusFilter,
  showAllDates,
  onToggleShowAllDates,
  onChangePeriod,
  onChangeSource,
  onChangeStatus,
  onResetFilters,
}: {
  periodFilter: PeriodFilter;
  sourceFilter: SourceFilter;
  statusFilter: StatusFilter;
  showAllDates: boolean;
  onToggleShowAllDates: () => void;
  onChangePeriod: (value: PeriodFilter) => void;
  onChangeSource: (value: SourceFilter) => void;
  onChangeStatus: (value: StatusFilter) => void;
  onResetFilters: () => void;
}) {
  const selectClass =
    "h-9 rounded-full border border-slate-200/80 bg-white px-3 text-[12px] font-bold text-slate-700 shadow-sm outline-none transition hover:border-indigo-200 focus:border-indigo-400 focus:ring-4 focus:ring-indigo-100";

  return (
    <div className="flex flex-wrap items-center gap-2">
      <label className="sr-only" htmlFor="timeline-period-filter">
        기간 필터
      </label>
      <select
        id="timeline-period-filter"
        aria-label="기간 필터"
        className={selectClass}
        value={periodFilter}
        onChange={(event) => onChangePeriod(event.target.value as PeriodFilter)}
      >
        <option value="all">전체 기간</option>
        <option value="7d">최근 7일</option>
        <option value="30d">최근 30일</option>
      </select>
      <label className="sr-only" htmlFor="timeline-source-filter">
        소스 필터
      </label>
      <select
        id="timeline-source-filter"
        aria-label="소스 필터"
        className={selectClass}
        value={sourceFilter}
        onChange={(event) => onChangeSource(event.target.value as SourceFilter)}
      >
        <option value="all">소스 전체</option>
        <option value="Slack">Slack</option>
        <option value="Gmail">Gmail</option>
        <option value="Drive">Drive</option>
        <option value="Calendar">Calendar</option>
      </select>
      <label className="sr-only" htmlFor="timeline-status-filter">
        상태 필터
      </label>
      <select
        id="timeline-status-filter"
        aria-label="상태 필터"
        className={selectClass}
        value={statusFilter}
        onChange={(event) => onChangeStatus(event.target.value as StatusFilter)}
      >
        <option value="all">상태 전체</option>
        <option value="승인">승인됨</option>
        <option value="완료">완료</option>
      </select>
      <button
        type="button"
        data-testid="timeline-date-density-toggle"
        aria-pressed={showAllDates}
        onClick={onToggleShowAllDates}
        className="inline-flex h-9 items-center gap-2 rounded-full border border-slate-200/80 bg-white px-3 text-[12px] font-extrabold text-slate-700 shadow-sm transition hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700"
      >
        {showAllDates ? "활동 있는 날짜만 보기" : "전체 날짜 보기"}
      </button>
      <button
        type="button"
        onClick={onResetFilters}
        className="inline-flex h-9 items-center gap-2 rounded-full border border-slate-200/80 bg-white px-3 text-[12px] font-extrabold text-slate-700 shadow-sm transition hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700"
      >
        <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
        필터 초기화
      </button>
    </div>
  );
}

function TimelineDateIndex({
  monthGroups,
  expandedMonthKeys,
  expandedDateKeys,
  onToggleMonth,
  onJumpDate,
}: {
  monthGroups: TimelineMonthGroupData[];
  expandedMonthKeys: Set<string>;
  expandedDateKeys: Set<string>;
  onToggleMonth: (monthKey: string) => void;
  onJumpDate: (dateKey: string) => void;
}) {
  return (
    <aside
      data-testid="timeline-date-index"
      className="hidden h-fit max-h-[calc(100vh-8.5rem)] overflow-y-auto rounded-[1.5rem] border border-white/80 bg-white/80 p-3 shadow-[0_20px_56px_rgba(15,23,42,0.06)] backdrop-blur xl:sticky xl:top-28 xl:block"
      aria-label="타임라인 날짜 인덱스"
    >
      <div className="space-y-3">
        {monthGroups.map((month) => (
          <div key={month.monthKey} className="rounded-[1.15rem] border border-slate-200/60 bg-white/70 p-2">
            <button
              type="button"
              data-testid={`timeline-month-nav-${month.monthKey}`}
              aria-expanded={expandedMonthKeys.has(month.monthKey)}
              className="flex w-full items-center justify-between gap-2 rounded-xl px-2 py-2 text-left transition hover:bg-indigo-50"
              onClick={() => onToggleMonth(month.monthKey)}
            >
              <span>
                <span className="block text-[12px] font-extrabold text-slate-950">{month.monthLabel}</span>
                <span className="text-[11px] font-bold text-slate-400">{month.dateGroups.length.toLocaleString()}일</span>
              </span>
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-extrabold text-slate-600">
                {monthGroupTotalCount(month).toLocaleString()}
              </span>
            </button>
            {expandedMonthKeys.has(month.monthKey) ? (
            <div className="mt-1 space-y-1">
              {month.dateGroups.map((dateGroup) => {
                const hasItems = dateGroup.items.length > 0;
                return (
                  <button
                    key={dateGroup.dateKey}
                    type="button"
                    data-testid={`timeline-date-index-${dateGroup.dateKey}`}
                    className={`flex w-full items-center justify-between rounded-xl px-2 py-1.5 text-left text-[12px] font-bold transition ${
                      expandedDateKeys.has(dateGroup.dateKey)
                        ? "bg-indigo-600 text-white shadow-sm"
                        : "text-slate-500 hover:bg-indigo-50 hover:text-indigo-700"
                    }`}
                    onClick={() => onJumpDate(dateGroup.dateKey)}
                  >
                    <span>{shortDateLabel(dateGroup.dateKey)}</span>
                    <span className={`rounded-full px-1.5 py-0.5 text-[10px] ${hasItems ? "bg-white/70 text-slate-600" : "bg-slate-200 text-slate-400"}`}>
                      {dateGroup.items.length}
                    </span>
                  </button>
                );
              })}
            </div>
            ) : null}
          </div>
        ))}
      </div>
    </aside>
  );
}

function MonthCollapsedSummary({ monthGroup, onExpand }: { monthGroup: TimelineMonthGroupData; onExpand: () => void }) {
  return (
    <div className="rounded-[1.5rem] border border-dashed border-slate-200 bg-white/65 p-4 shadow-sm">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-extrabold text-slate-950">{monthGroup.monthLabel}</p>
          <p className="mt-1 text-sm font-medium text-slate-500">
            {monthGroup.dateGroups.length.toLocaleString()}일 · {monthGroupTotalCount(monthGroup).toLocaleString()}건이 접혀 있습니다.
          </p>
        </div>
        <button
          type="button"
          className="inline-flex h-9 items-center justify-center rounded-full border border-slate-200 bg-white px-3 text-xs font-extrabold text-indigo-700 shadow-sm transition hover:bg-indigo-50"
          onClick={onExpand}
        >
          펼치기
        </button>
      </div>
    </div>
  );
}

function TimelineDateGroup({
  group,
  isExpanded,
  isShowingAll,
  selectedHistoryId,
  onToggleDate,
  onToggleMore,
  onSelectHistory,
}: {
  group: TimelineDateGroupData;
  isExpanded: boolean;
  isShowingAll: boolean;
  selectedHistoryId?: string;
  onToggleDate: (dateKey: string) => void;
  onToggleMore: (dateKey: string) => void;
  onSelectHistory: (historyId: string) => void;
}) {
  const visibleItems = isShowingAll ? group.items : group.items.slice(0, 3);
  const hiddenCount = group.items.length - visibleItems.length;

  return (
    <section id={`timeline-date-${group.dateKey}`} className="scroll-mt-28 rounded-[1.5rem] border border-white/80 bg-white/80 p-4 shadow-sm">
      <button
        type="button"
        aria-expanded={isExpanded}
        className="group flex w-full items-center justify-between gap-3 rounded-2xl text-left transition"
        onClick={() => onToggleDate(group.dateKey)}
      >
        <span className="flex min-w-0 items-center gap-3">
          <span className={`grid h-10 w-10 shrink-0 place-items-center rounded-2xl ${isExpanded ? "bg-indigo-50 text-indigo-600" : "bg-slate-100 text-slate-400"}`}>
            <CalendarDays className="h-4 w-4" aria-hidden="true" />
          </span>
          <span className="min-w-0">
            <h3 className="truncate text-[15px] font-extrabold text-slate-950">{group.dateLabel}</h3>
            <p className="mt-0.5 text-xs font-semibold text-slate-500">{sourceSummaryLabel(group.items)}</p>
          </span>
        </span>
        <span className="flex shrink-0 items-center gap-2">
          <span className="rounded-full bg-indigo-50 px-2.5 py-1 text-[11px] font-extrabold text-indigo-700">
            {group.items.length > 0 ? `${group.items.length.toLocaleString()}건` : "활동 없음"}
          </span>
          <ChevronDown
            className={`h-4 w-4 text-slate-400 transition-transform ${isExpanded ? "rotate-180" : ""}`}
            aria-hidden="true"
          />
        </span>
      </button>
      {isExpanded ? (
        <div className="mt-3 space-y-2">
          {visibleItems.length > 0 ? visibleItems.map((item) => (
            <TimelineEventRow
              key={item.id}
              item={item}
              isSelected={selectedHistoryId === item.id}
              onSelectHistory={onSelectHistory}
            />
          )) : (
            <div className="rounded-lg border border-dashed border-line bg-surface-soft p-4 text-[13px] font-semibold text-muted">
              이 날짜의 활동이 없습니다.
            </div>
          )}
          {group.items.length > 3 ? (
            <button
              type="button"
              className="ml-1 inline-flex items-center gap-1.5 rounded-md px-1 py-1 text-[12px] font-extrabold text-[var(--primary-dark)] hover:underline"
              onClick={() => onToggleMore(group.dateKey)}
            >
              {isShowingAll ? (
                <>
                  <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                  접기
                </>
              ) : (
                <>
                  {hiddenCount.toLocaleString()}건 더 보기
                  <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
                </>
              )}
            </button>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function TimelineEventRow({
  item,
  isSelected,
  onSelectHistory,
}: {
  item: TimelineHistory;
  isSelected: boolean;
  onSelectHistory: (historyId: string) => void;
}) {
  return (
    <article className="grid gap-3 rounded-2xl border border-slate-200/70 bg-white p-4 shadow-sm transition hover:-translate-y-0.5 hover:border-indigo-200 hover:shadow-md sm:grid-cols-[40px_minmax(0,1fr)_auto] sm:items-center">
      <span className={`grid h-10 w-10 place-items-center rounded-2xl ${sourceIconClass(item.source)}`}>
        <TimelineSourceIcon item={item} />
      </span>
      <div className="min-w-0">
        <h3 className="line-clamp-1 text-[14px] font-extrabold leading-5 text-slate-950">{item.title}</h3>
        <p className="mt-1 line-clamp-1 text-[12px] font-medium text-slate-500">{item.preview}</p>
      </div>
      <div className="flex flex-wrap items-center gap-2 sm:justify-end">
        <time className="text-[12px] font-bold text-slate-400">{item.time}</time>
        <span className={`rounded-full border px-2.5 py-1 text-[11px] font-extrabold ${sourceBadgeClass(item.source)}`}>
          {item.source}
        </span>
        <span className={`rounded-full border px-2.5 py-1 text-[11px] font-extrabold ${statusChipClass(item.status)}`}>
          {statusLabel(item.status)}
        </span>
        <AutoReviewBadge resolutionSource={item.resolutionSource} />
        <button
          type="button"
          aria-pressed={isSelected}
          className={`icon-button small shrink-0 ${isSelected ? "active" : ""}`}
          aria-label={`Open ${item.title}`}
          onClick={() => onSelectHistory(item.id)}
        >
          <Eye className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>
    </article>
  );
}

const sourceIconImages: Record<string, ImageProps["src"]> = {
  Slack: slackIcon,
  Gmail: gmailIcon,
  Drive: driveIcon,
  Calendar: calendarIcon,
};

function TimelineSourceIcon({ item }: { item: TimelineHistory }) {
  const src = sourceIconImages[item.source] ?? todoIcon;

  return (
    <Image
      src={src}
      alt=""
      width={18}
      height={18}
      className="h-4.5 w-4.5 object-contain"
      aria-hidden="true"
    />
  );
}


function timelineHistoryFromProjectItem(item: ProjectTimelineItem): TimelineHistory {
  const occurredAt = item.occurred_at || item.created_at;
  const source = sourceFromLinks(item.source_links);
  const status: TimelineStatus = item.completed_at || isPastCalendarTimeline(source, occurredAt) ? "완료" : "승인";
  return {
    id: item.id,
    itemType: item.item_type,
    createdAt: occurredAt,
    time: formatTime(occurredAt),
    source,
    title: item.title,
    summary: item.summary,
    history: item.summary || "승인된 타임라인 요약이 없습니다.",
    status,
    resolutionSource: item.resolution_source,
    sourceUrl: item.source_links[0] ?? "",
    preview: previewForProjectItem(item, source),
    snippets: item.source_snippets.map((snippet) => ({
      author: `${source} evidence · ${item.evidence_reason || "승인된 프로젝트 근거"}`,
      body: snippet,
      time: formatTime(occurredAt),
    })),
  };
}

function groupHistoriesByDate(histories: TimelineHistory[], includeEmptyDates = false): TimelineDateGroupData[] {
  const groups = new Map<string, TimelineHistory[]>();
  for (const history of [...histories].sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt))) {
    const key = dateKeyForValue(history.createdAt);
    groups.set(key, [...(groups.get(key) ?? []), history]);
  }

  if (includeEmptyDates && groups.size > 0) {
    const keys = Array.from(groups.keys()).sort();
    const start = dateFromKey(keys[0]);
    const end = dateFromKey(keys[keys.length - 1]);
    for (const date = end; date >= start; date.setDate(date.getDate() - 1)) {
      const key = dateKeyForValue(date.toISOString());
      if (!groups.has(key)) groups.set(key, []);
    }
  }

  return Array.from(groups.entries())
    .sort(([left], [right]) => right.localeCompare(left))
    .map(([dateKey, items]) => ({
      dateKey,
      dateLabel: formatDateFromKey(dateKey),
      monthKey: dateKey.slice(0, 7),
      monthLabel: formatMonthFromKey(dateKey),
      items,
    }));
}

function groupDateGroupsByMonth(groups: TimelineDateGroupData[]): TimelineMonthGroupData[] {
  const monthGroups = new Map<string, TimelineDateGroupData[]>();
  for (const group of groups) {
    monthGroups.set(group.monthKey, [...(monthGroups.get(group.monthKey) ?? []), group]);
  }
  return Array.from(monthGroups.entries()).map(([monthKey, dateGroups]) => ({
    monthKey,
    monthLabel: dateGroups[0]?.monthLabel ?? monthKey,
    dateGroups,
  }));
}

function defaultExpandedMonthKeys(monthGroups: TimelineMonthGroupData[]) {
  const firstActive = monthGroups.find((month) => monthGroupTotalCount(month) > 0)?.monthKey;
  return firstActive ? new Set([firstActive]) : new Set<string>();
}

function defaultExpandedDateKeys(groups: TimelineDateGroupData[]) {
  const firstActive = groups.find((group) => group.items.length > 0)?.dateKey;
  return firstActive ? new Set([firstActive]) : new Set<string>();
}

function monthGroupTotalCount(monthGroup: TimelineMonthGroupData) {
  return monthGroup.dateGroups.reduce((sum, group) => sum + group.items.length, 0);
}

function sourceSummaryLabel(items: TimelineHistory[]) {
  if (items.length === 0) return "활동 없음";
  const counts = items.reduce<Record<string, number>>((acc, item) => {
    acc[item.source] = (acc[item.source] ?? 0) + 1;
    return acc;
  }, {});
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 2)
    .map(([source, count]) => `${source} ${count.toLocaleString()}건`)
    .join(" · ");
}

function firstProjectWithHistories(projects: ProjectTimeline[]) {
  return projects.find((project) => project.histories.length > 0) ?? projects[0];
}

function sourceFromLinks(links: string[]): TimelineSource {
  const joined = links.join(" ").toLowerCase();
  if (joined.includes("slack")) return "Slack";
  if (joined.includes("gmail") || joined.includes("mail")) return "Gmail";
  if (joined.includes("drive") || joined.includes("docs.google")) return "Drive";
  if (joined.includes("calendar")) return "Calendar";
  return "Source";
}

function isPastCalendarTimeline(source: TimelineSource, occurredAt: string) {
  if (source !== "Calendar") return false;
  const timestamp = Date.parse(occurredAt);
  if (Number.isNaN(timestamp)) return false;
  return timestamp < Date.now();
}

function sourceBadgeClass(source: TimelineSource) {
  if (source === "Drive") return "border-blue-100 bg-blue-50 text-blue-700";
  if (source === "Slack") return "border-violet-100 bg-violet-50 text-violet-700";
  if (source === "Gmail") return "border-rose-100 bg-rose-50 text-rose-700";
  if (source === "Calendar") return "border-emerald-100 bg-emerald-50 text-emerald-700";
  return "border-slate-200 bg-slate-50 text-slate-600";
}

function sourceIconClass(source: TimelineSource) {
  if (source === "Drive") return "bg-blue-50 text-blue-700";
  if (source === "Slack") return "bg-violet-50 text-violet-700";
  if (source === "Gmail") return "bg-rose-50 text-rose-700";
  if (source === "Calendar") return "bg-emerald-50 text-emerald-700";
  return "bg-slate-50 text-slate-600";
}

function statusChipClass(status: TimelineStatus) {
  if (status === "승인") return "border-emerald-100 bg-emerald-50 text-emerald-700";
  return "border-blue-100 bg-blue-50 text-blue-700";
}

function statusLabel(status: TimelineStatus) {
  if (status === "승인") return "승인됨";
  return status;
}

function previewForProjectItem(item: ProjectTimelineItem, source: TimelineSource) {
  const snippet = item.source_snippets.find((value) => value.trim().length > 0);
  if (snippet) return snippet;
  if (item.summary) return item.summary;
  return `${source} 근거가 연결된 승인 타임라인 항목입니다.`;
}

function filterHistories(
  histories: TimelineHistory[],
  periodFilter: PeriodFilter,
  sourceFilter: SourceFilter,
  statusFilter: StatusFilter,
) {
  const now = Date.now();
  return histories.filter((history) => {
    if (sourceFilter !== "all" && history.source !== sourceFilter) return false;
    if (statusFilter !== "all" && history.status !== statusFilter) return false;
    if (periodFilter === "all") return true;
    const timestamp = Date.parse(history.createdAt);
    if (Number.isNaN(timestamp)) return true;
    const days = periodFilter === "7d" ? 7 : 30;
    return now - timestamp <= days * 24 * 60 * 60 * 1000;
  });
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--:--";
  return new Intl.DateTimeFormat("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Asia/Seoul",
  }).format(date);
}

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "날짜 미상";
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "Asia/Seoul",
  }).format(date);
}

function dateKeyForValue(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "0000-00-00";
  return date.toLocaleDateString("en-CA", { timeZone: "Asia/Seoul" });
}

function dateFromKey(dateKey: string) {
  return new Date(`${dateKey}T00:00:00+09:00`);
}

function formatDateFromKey(dateKey: string) {
  return formatDate(`${dateKey}T00:00:00+09:00`);
}

function formatMonthFromKey(dateKey: string) {
  const date = dateFromKey(dateKey);
  if (Number.isNaN(date.getTime())) return "날짜 미상";
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "long",
    timeZone: "Asia/Seoul",
  }).format(date);
}

function shortDateLabel(dateKey: string) {
  const date = dateFromKey(dateKey);
  if (Number.isNaN(date.getTime())) return dateKey;
  return new Intl.DateTimeFormat("ko-KR", {
    month: "numeric",
    day: "numeric",
    timeZone: "Asia/Seoul",
  }).format(date);
}
