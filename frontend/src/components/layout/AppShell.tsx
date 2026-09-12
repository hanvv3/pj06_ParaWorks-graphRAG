"use client";

import {
  Bell,
  Bot,
  CalendarClock,
  CircleHelp,
  Database,
  FolderKanban,
  GitBranch,
  Grid2X2,
  Inbox,
  Search,
  Settings,
} from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { FormEvent, type ReactNode, useEffect, useMemo, useState } from "react";
import { apiGet, apiPost, clearStoredDemoUserId } from "@/lib/api/client";
import { REVIEW_QUEUE_UPDATED_EVENT } from "@/lib/reviewQueueEvents";
import type { AuthUserResponse, DashboardResponse, DemoUser, NotificationsResponse } from "@/lib/api/types";
import { LanguageProvider } from "@/lib/i18n/LanguageProvider";
import { ephemeralSearchHandoff } from "@/lib/assistant/searchHandoff";

/**
 * 네비게이션 아이템의 타입 정의
 */
type NavItem = {
  href: string;
  label: string;
  icon: typeof Grid2X2;
  badge?: number;
  requiredRole?: "admin";
};

/**
 * 사이드바에 표시될 네비게이션 그룹 및 아이템 설정
 */
const navGroups: { items: NavItem[] }[] = [
  {
    items: [
      { href: "/dashboard", label: "대시보드", icon: Grid2X2 },
      { href: "/projects", label: "프로젝트", icon: FolderKanban },
      { href: "/review", label: "검토사항", icon: Inbox },
      { href: "/timeline", label: "타임라인", icon: GitBranch },
    ],
  },
  {
    items: [
      { href: "/search", label: "AI 비서", icon: Bot },
      {
        href: "/agent-runs",
        label: "에이전트 실행 기록",
        icon: CalendarClock,
        requiredRole: "admin",
      },
      { href: "/integrations", label: "연동 관리", icon: Database },
      { href: "/notifications", label: "알림", icon: Bell },
    ],
  },
  {
    items: [
      { href: "/admin", label: "관리자 콘솔", icon: Settings, requiredRole: "admin" },
    ],
  },
];

/**
 * AppShell 컴포넌트: 어플리케이션의 공통 레이아웃을 담당합니다.
 * 다국어 지원을 위한 LanguageProvider와 내부 레이아웃인 ShellContent를 래핑합니다.
 */
export function AppShell({ children }: { children: ReactNode }) {
  return (
    <LanguageProvider>
      <ShellContent>{children}</ShellContent>
    </LanguageProvider>
  );
}

/**
 * ShellContent 컴포넌트: 실제 사이드바, 헤더, 메인 콘텐츠 영역을 렌더링합니다.
 * 인증 상태 확인, 배지 카운트 조회, 네비게이션 필터링 등의 로직을 포함합니다.
 */
function ShellContent({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [currentUser, setCurrentUser] = useState<DemoUser | null>(null);
  const [badgeCounts, setBadgeCounts] = useState<{ review?: number; notifications?: number }>({});

  useEffect(() => {
    function refreshBadgeCounts() {
      apiGet<DashboardResponse>("/api/v1/dashboard")
        .then((res) => setBadgeCounts((prev) => ({ ...prev, review: res.pending_review_count || 0 })))
        .catch(() => undefined);

      apiGet<NotificationsResponse>("/api/v1/notifications")
        .then((res) => setBadgeCounts((prev) => ({ ...prev, notifications: res.counts?.total || 0 })))
        .catch(() => undefined);
    }

    window.addEventListener(REVIEW_QUEUE_UPDATED_EVENT, refreshBadgeCounts);
    return () => window.removeEventListener(REVIEW_QUEUE_UPDATED_EVENT, refreshBadgeCounts);
  }, []);

  // 페이지 로드 시 및 경로 변경 시 인증 상태 및 배지 카운트 확인
  useEffect(() => {
    if (pathname === "/login" || pathname.startsWith("/login/")) {
      return;
    }

    let active = true;
    
    // 현재 로그인된 사용자 정보 조회
    apiGet<AuthUserResponse>("/api/v1/auth/me")
      .then((result) => {
        if (active) {
          setCurrentUser(result.user);
        }
      })
      .catch(() => {
        if (active) {
          setCurrentUser(null);
          router.push("/login"); // 인증 실패 시 로그인 페이지로 이동
        }
      });

    // 대시보드 배지(검토 대기 건수) 로드
    apiGet<DashboardResponse>("/api/v1/dashboard")
      .then((res) => active && setBadgeCounts((prev) => ({ ...prev, review: res.pending_review_count || 0 })))
      .catch(() => {});
      
    // 알림 배지 카운트 로드
    apiGet<NotificationsResponse>("/api/v1/notifications")
      .then((res) => active && setBadgeCounts((prev) => ({ ...prev, notifications: res.counts?.total || 0 })))
      .catch(() => {});

    return () => {
      active = false;
    };
  }, [pathname, router]);

  /**
   * 사용자 권한에 따라 노출할 네비게이션 메뉴를 필터링하고 배지 숫자를 적용합니다.
   */
  const visibleNavGroups = useMemo(
    () =>
      navGroups
        .map((group) => ({
          items: group.items
            .filter((item) => !item.requiredRole || currentUser?.role === item.requiredRole)
            .map((item) => {
              if (item.href === "/review") return { ...item, badge: badgeCounts.review || undefined };
              if (item.href === "/notifications") return { ...item, badge: badgeCounts.notifications || undefined };
              return item;
            }),
        }))
        .filter((group) => group.items.length > 0),
    [currentUser?.role, badgeCounts],
  );

  /**
   * 하단 사용자 프로필 영역에 표시할 정보를 계산합니다.
   */
  const accountDisplay = useMemo(() => {
    const isLoggedIn = !!currentUser;
    const name = currentUser?.name ?? "로그인이 필요합니다";
    const roleLabels: Record<string, string> = {
      admin: "관리자",
      manager: "매니저",
      reviewer: "리뷰어",
      employee: "멤버",
    };
    const role = currentUser ? (roleLabels[currentUser.role] ?? currentUser.role) : "여기를 클릭해 로그인하세요";
    const initial = isLoggedIn ? (currentUser?.name?.trim().charAt(0).toUpperCase() || "?") : "!";
    const avatarUrl = currentUser?.avatar_url ?? null;
    return { name, role, initial, avatarUrl, isLoggedIn };
  }, [currentUser]);

  // 로그인 페이지에서는 셸 없이 콘텐츠만 렌더링
  if (pathname === "/login" || pathname.startsWith("/login/")) {
    return <>{children}</>;
  }

  /**
   * 상단 검색창 제출 시 AI 비서(검색) 페이지로 이동
   */
  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const rawQuery = String(formData.get("q") ?? "");
    ephemeralSearchHandoff.put(rawQuery);
    router.push("/search");
  }

  /**
   * 로그아웃 처리
   */
  async function logout() {
    setAccountMenuOpen(false);
    try {
      await apiPost<{ status: string }>("/api/v1/auth/logout");
    } finally {
      clearStoredDemoUserId();
      setCurrentUser(null);
      router.push("/login");
    }
  }

  return (
    <div className="min-h-screen bg-app text-ink">
      {/* 데스크탑 사이드바 */}
      <aside className="app-sidebar fixed inset-y-0 left-0 z-30 hidden w-[240px] border-r border-white/70 bg-white/85 px-5 py-7 shadow-[18px_0_48px_rgba(15,23,42,0.05)] backdrop-blur lg:flex lg:flex-col">
        {/* 로고 영역 */}
        <Link href="/dashboard" className="flex items-start gap-3">
          <span className="brand-logo" aria-hidden="true">
            <Image src="/assets/paraworks-logo-icon.png" alt="" width={37} height={37} />
          </span>
          <span className="min-w-0 pt-0.5">
            <span className="brand-wordmark block text-[19px] leading-6 text-[var(--primary-dark)]">paraworks</span>
          </span>
        </Link>

        {/* 네비게이션 메뉴 */}
        <nav className="app-sidebar-nav mt-6 min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
          {visibleNavGroups.map((group, groupIndex) => (
            <div key={groupIndex} className="border-t border-line py-4 first:border-t-0 first:pt-0">
              <div className="space-y-2">
                {group.items.map((item) => {
                  const Icon = item.icon;
                  const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
                  return (
                    <Link key={item.href} href={item.href} className={`sidebar-link ${active ? "active" : ""}`}>
                      <Icon className="h-[18px] w-[18px]" aria-hidden="true" />
                      <span className="min-w-0 flex-1 truncate">{item.label}</span>
                      {item.badge ? <span className="nav-badge">{item.badge}</span> : null}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* 하단 계정/프로필 영역 */}
        <div className="relative mt-4 flex items-center justify-between rounded-2xl border border-[#e7ebf4] bg-white p-2.5 shadow-[0_18px_38px_rgba(15,23,42,0.07)]">
          <button
            type="button"
            className={`flex min-w-0 flex-1 items-center gap-2 p-1 text-left outline-none transition-opacity hover:opacity-80 ${
              !accountDisplay.isLoggedIn ? "animate-pulse" : ""
            }`}
            onClick={() => {
              if (!accountDisplay.isLoggedIn) {
                router.push("/login");
              } else {
                setAccountMenuOpen((open) => !open);
              }
            }}
          >
            <span className={`avatar-photo ${!accountDisplay.isLoggedIn ? "bg-[var(--primary-soft)]" : ""}`} aria-hidden="true">
              {accountDisplay.avatarUrl ? (
                <Image src={accountDisplay.avatarUrl} alt="" width={34} height={34} />
              ) : (
                <span className={!accountDisplay.isLoggedIn ? "text-[var(--primary-dark)]" : ""}>
                  {accountDisplay.initial}
                </span>
              )}
            </span>
            <span className="min-w-0 flex-1">
              <span className={`block truncate text-[13px] font-bold ${accountDisplay.isLoggedIn ? "text-ink" : "text-[var(--primary-dark)]"}`}>
                {accountDisplay.name}
              </span>
              <span className="block truncate text-[11px] text-muted">{accountDisplay.role}</span>
            </span>
          </button>
          
          <button
            type="button"
            className="icon-button small shrink-0"
            aria-label={"계정 메뉴"}
            aria-expanded={accountMenuOpen}
            onClick={() => setAccountMenuOpen((open) => !open)}
          >
            <Settings className="h-4 w-4" aria-hidden="true" />
          </button>

          {/* 계정 드롭다운 메뉴 */}
          {accountMenuOpen ? (
            <div className="absolute bottom-full right-0 z-40 mb-2 w-44 rounded-lg border border-line bg-[var(--glass-elevated)] p-1 shadow-lg">
              {accountDisplay.isLoggedIn ? (
                <>
                  <button
                    type="button"
                    className="w-full rounded-md px-3 py-2 text-left text-[13px] font-bold text-ink hover:bg-[#f3f7fd]"
                    onClick={() => {
                      setAccountMenuOpen(false);
                      router.push("/account");
                    }}
                  >
                    내 계정 정보
                  </button>
                  <div className="my-1 border-t border-line" />
                  <button
                    type="button"
                    className="w-full rounded-md px-3 py-2 text-left text-[13px] font-bold text-red-600 hover:bg-red-50"
                    onClick={() => void logout()}
                  >
                    로그아웃
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  className="w-full rounded-md px-3 py-2 text-left text-[13px] font-bold text-[var(--primary-dark)] hover:bg-[var(--primary-soft)]"
                  onClick={() => {
                    setAccountMenuOpen(false);
                    router.push("/login");
                  }}
                >
                  로그인하기
                </button>
              )}
            </div>
          ) : null}
        </div>
      </aside>

      <div className="lg:pl-[240px]">
        {/* 상단 헤더 */}
        <header className="sticky top-0 z-20 bg-[#f3f5fa]/90 px-4 py-5 backdrop-blur md:px-8 lg:px-8">
          <div className="mx-auto flex max-w-[1500px] items-center justify-between gap-4">
            {/* 모바일 로고 */}
            <Link href="/dashboard" className="flex items-center gap-2 lg:hidden">
              <span className="brand-logo small" aria-hidden="true">
                <Image src="/assets/paraworks-logo-icon.png" alt="" width={31} height={31} />
              </span>
              <span className="brand-wordmark text-[var(--primary-dark)]">paraworks</span>
            </Link>

            {/* 통합 검색창 */}
            <form onSubmit={submitSearch} className="top-search ml-auto min-w-0 flex-1 md:max-w-[620px]">
              <button type="submit" className="top-search-icon-button" aria-label="AI 비서에게 질문">
                <Search className="h-4 w-4" aria-hidden="true" />
              </button>
              <input
                name="q"
                className="min-w-0 flex-1 bg-transparent text-[13px] outline-none placeholder:text-[#667085]"
                placeholder={"검색어를 입력하세요"}
                aria-label={"회사 메모리 검색"}
              />
            </form>

            {/* 헤더 우측 유틸리티 버튼 */}
            <div className="flex shrink-0 items-center gap-3">
              <button type="button" className="icon-button" aria-label={"도움말"}>
                <CircleHelp className="h-[18px] w-[18px]" aria-hidden="true" />
              </button>
              <Link href="/notifications" className="icon-button notification-button" aria-label={"알림"}>
                <Bell className="h-[18px] w-[18px]" aria-hidden="true" />
                {badgeCounts.notifications ? <span>{badgeCounts.notifications}</span> : null}
              </Link>
              <button
                type="button"
                className="hidden items-center gap-2 rounded-full border border-[#e0e6f0] bg-white py-1.5 pl-1.5 pr-3 shadow-[0_12px_30px_rgba(15,23,42,0.06)] md:flex"
                onClick={() => {
                  if (accountDisplay.isLoggedIn) {
                    router.push("/account");
                  } else {
                    router.push("/login");
                  }
                }}
                aria-label="사용자 프로필"
              >
                <span className="avatar-photo" aria-hidden="true">
                  {accountDisplay.avatarUrl ? (
                    <Image src={accountDisplay.avatarUrl} alt="" width={34} height={34} />
                  ) : (
                    <span>{accountDisplay.initial}</span>
                  )}
                </span>
                <span className="max-w-[8rem] truncate text-[13px] font-extrabold text-ink">{accountDisplay.name}</span>
              </button>
            </div>
          </div>
        </header>

        {/* 메인 콘텐츠 영역 */}
        <main className="mx-auto w-full max-w-[1500px] px-4 pb-10 pt-1 md:px-8 lg:px-8">{children}</main>
      </div>
    </div>
  );
}
