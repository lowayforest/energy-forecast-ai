import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import * as echarts from 'echarts/core'
import { LineChart, BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import raw from './demo-data.json'
import futureRaw from './future-data.json'
import './style.css'

type Day = {
  date: string
  actual: number
  physics: number
  ai: number | null
  aiReason: string | null
  aiFactor: number | null
  ghi: number[]
  temp: number[]
  runUtc: string
  adaptiveFactor: number
  observedWeatherPhysics: number
}
type Source = { name: string; use: string; url: string; provider: string }
type FutureDay = { date: string; actual: null; physics: number; ai: number | null; aiFactor: number | null; aiConfidence: string | null; aiReason: string | null; ghi: number[]; temp: number[]; runUtc: string }
type FutureData = { meta: { period: string; station: string; kind: string; isCurrentForecast: boolean; lastActualDate: string; calibrationFrozenThrough: string; calibrationPeriod: string; weather: string; aiModel: string; aiStatus: string }; metrics: { days: number; actualTotal: null; physicsTotal: number; aiTotal: number | null; mae: null; mape: null }; daily: FutureDay[] }
type Data = {
  meta: { title: string; site: string; region: string; period: string; ratedKw: number; calibration: string; calibrationDays: number; siteScale: number; aiModel: string; aiStatus: string; disclaimer: string }
  metrics: { days: number; completeMonth: boolean; actualTotal: number; physicsTotal: number; aiTotal: number | null; physicsMae: number | null; aiMae: number | null; aiEvaluatedDays: number }
  daily: Day[]
  sources: Source[]
  method: { step: string; title: string; text: string }[]
}
const data = raw as Data
const future = futureRaw as FutureData
const fmt = (n: number | null | undefined, digits = 1) => n == null ? '—' : n.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
echarts.use([LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer])

function Chart({ setSelected }: { setSelected: (d: string) => void }) {
  const element = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!element.current) return
    const chart = echarts.init(element.current)
    const days = data.daily
    chart.setOption({
      animationDuration: 600,
      grid: { left: 44, right: 22, top: 44, bottom: 44 },
      tooltip: { trigger: 'axis', backgroundColor: '#172b2d', borderColor: '#172b2d', textStyle: { color: '#fff' }, valueFormatter: (value: number) => value == null ? '—' : `${fmt(value)} kWh` },
      legend: { top: 0, right: 8, itemWidth: 12, itemHeight: 6, textStyle: { color: '#65737a', fontSize: 12 }, data: ['实际发电量', '物理预测', ...(data.metrics.aiEvaluatedDays ? ['豆包修正'] : [])] },
      xAxis: { type: 'category', boundaryGap: false, data: days.map(d => d.date.slice(8)), axisLine: { lineStyle: { color: '#d5dedc' } }, axisTick: { show: false }, axisLabel: { color: '#819092', fontSize: 11, interval: 2, formatter: (v: string) => `${Number(v)}日` } },
      yAxis: { type: 'value', name: 'kWh', nameTextStyle: { color: '#849195', align: 'right' }, min: 0, splitLine: { lineStyle: { color: '#edf1ef' } }, axisLabel: { color: '#819092' } },
      series: [
        { name: '实际发电量', type: 'line', smooth: 0.22, symbol: 'circle', symbolSize: 5, lineStyle: { width: 3, color: '#159879' }, itemStyle: { color: '#159879' }, areaStyle: { color: 'rgba(21,152,121,.10)' }, data: days.map(d => d.actual) },
        { name: '物理预测', type: 'line', smooth: 0.22, symbol: 'circle', symbolSize: 5, lineStyle: { width: 2.5, color: '#e5a657' }, itemStyle: { color: '#e5a657' }, data: days.map(d => d.physics) },
        ...(data.metrics.aiEvaluatedDays ? [{ name: '豆包修正', type: 'line', smooth: 0.22, symbol: 'circle', symbolSize: 5, lineStyle: { width: 2.5, color: '#6f79ba' }, itemStyle: { color: '#6f79ba' }, data: days.map(d => d.ai) }] : []),
      ],
    })
    chart.on('click', (params) => { if (typeof params.dataIndex === 'number') setSelected(days[params.dataIndex].date) })
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(element.current)
    return () => { observer.disconnect(); chart.dispose() }
  }, [setSelected])
  return <div ref={element} className="chart" aria-label="每日实际发电量与预测结果折线图" />
}

function WeatherChart({ day }: { day: Day }) {
  const element = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!element.current) return
    const chart = echarts.init(element.current)
    chart.setOption({
      animationDuration: 400,
      grid: { left: 34, right: 18, top: 18, bottom: 28 },
      tooltip: { trigger: 'axis', valueFormatter: (v: number) => `${fmt(v, 0)}` },
      xAxis: { type: 'category', data: ['05–08', '08–11', '11–14', '14–17', '17–20'], axisLine: { lineStyle: { color: '#dbe4e0' } }, axisTick: { show: false }, axisLabel: { color: '#82908c', fontSize: 10 } },
      yAxis: { type: 'value', min: 0, splitLine: { lineStyle: { color: '#f0f3f1' } }, axisLabel: { color: '#82908c', fontSize: 10 } },
      series: [{ type: 'bar', data: day.ghi, barWidth: '40%', itemStyle: { color: '#e6b469', borderRadius: [4, 4, 0, 0] } }],
    })
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(element.current)
    return () => { observer.disconnect(); chart.dispose() }
  }, [day])
  return <div ref={element} className="weather-chart" aria-label="GFS分时太阳辐照度预报柱状图" />
}

function FutureChart() {
  const element = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!element.current) return
    const chart = echarts.init(element.current)
    chart.setOption({
      animationDuration: 550,
      grid: { left: 44, right: 22, top: 42, bottom: 42 },
      tooltip: { trigger: 'axis', backgroundColor: '#172b2d', borderColor: '#172b2d', textStyle: { color: '#fff' }, valueFormatter: (v: number) => v == null ? '—' : `${fmt(v)} kWh` },
      legend: { top: 0, right: 8, itemWidth: 12, itemHeight: 6, textStyle: { color: '#65737a', fontSize: 12 }, data: ['物理预测', '豆包修正'] },
      xAxis: { type: 'category', boundaryGap: false, data: future.daily.map(d => d.date.slice(8)), axisLine: { lineStyle: { color: '#d5dedc' } }, axisTick: { show: false }, axisLabel: { color: '#819092', fontSize: 11, interval: 2, formatter: (v: string) => `${Number(v)}日` } },
      yAxis: { type: 'value', name: 'kWh', nameTextStyle: { color: '#849195', align: 'right' }, min: 0, splitLine: { lineStyle: { color: '#edf1ef' } }, axisLabel: { color: '#819092' } },
      series: [
        { name: '物理预测', type: 'line', smooth: 0.22, symbol: 'circle', symbolSize: 5, lineStyle: { width: 2.5, color: '#e5a657' }, itemStyle: { color: '#e5a657' }, data: future.daily.map(d => d.physics) },
        { name: '豆包修正', type: 'line', smooth: 0.22, symbol: 'circle', symbolSize: 5, lineStyle: { width: 2.5, color: '#6f79ba' }, itemStyle: { color: '#6f79ba' }, data: future.daily.map(d => d.ai) },
      ],
    })
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(element.current)
    return () => { observer.disconnect(); chart.dispose() }
  }, [])
  return <div ref={element} className="chart" aria-label="2024年1月无实测物理预测与豆包预测折线图" />
}

function App() {
  const [selected, setSelected] = useState(data.daily.at(-1)?.date || '')
  const day = data.daily.find(d => d.date === selected) || data.daily.at(-1)
  const complete = data.metrics.completeMonth
  const aiReady = data.metrics.aiEvaluatedDays > 0
  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-icon">✳</div><div><strong>SolarScope</strong><small>光伏预测实验平台</small></div></div>
      <div className="nav-caption">研究工作台</div>
      <nav>
        <a className="nav-item active" href="#overview"><span>◫</span> 预测概览</a>
        <a className="nav-item" href="#daily"><span>▥</span> 日度分析</a>
        <a className="nav-item" href="#future"><span>↗</span> 无实测新预测</a>
        <a className="nav-item" href="#method"><span>◇</span> 方法流程</a>
        <a className="nav-item" href="#sources"><span>☷</span> 数据来源</a>
      </nav>
      <div className="sidebar-bottom"><span className="live-dot" /> 公开数据研究演示 <small>验证站点 · HKUST SQ1</small></div>
    </aside>
    <main>
      <header className="topbar"><div className="crumb">研究工作台 <span>/</span> 预测概览</div><div className="top-right"><span className="period-pill">◷ &nbsp;历史回测 + 数据集外预测</span><span className="avatar">光</span></div></header>
      <div className="content">
        <section className="hero" id="overview"><div><div className="eyebrow"><span className="eyebrow-line" /> PHYSICS + AI FORECASTING</div><h1>光伏发电量预测<span className="headline-dot">.</span></h1><p>以已发布的天气预报驱动物理模型，结合豆包大模型修正，检验日发电量与月度累计表现。</p><div className="hero-tags"><span>● &nbsp;站点：HKUST SQ1</span><span>◉ &nbsp;额定功率：27.6 kW</span><span>▣ &nbsp;物理校准：2021–2022</span></div></div><div className="sun-graphic"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="sun-disc">☀</div><div className="solar-grid"/></div></section>
        <div className="notice"><span className="notice-icon">i</span><span>{data.meta.disclaimer}</span></div>
        <div className="section-heading"><div><span className="section-kicker">MONTHLY OVERVIEW</span><h2>月度表现</h2></div><span className="muted">{complete ? '2023 年 8 月 · 31 天完整回测' : `2023 年 8 月 · 已处理 ${data.metrics.days}/31 天`}</span></div>
        <div className="stat-grid">
          <div className="stat-card actual"><div className="stat-top"><span>实际发电量累计</span><span className="stat-icon">▤</span></div><div className="stat-value">{fmt(data.metrics.actualTotal, 0)} <small>kWh</small></div><div className="stat-foot">{complete ? '8 月实际月发电量' : '已处理日期累计值'}</div></div>
          <div className="stat-card physics"><div className="stat-top"><span>物理预测累计</span><span className="stat-icon">☼</span></div><div className="stat-value">{fmt(data.metrics.physicsTotal, 0)} <small>kWh</small></div><div className="stat-foot">逐日次日预测汇总 · MAE {fmt(data.metrics.physicsMae)} kWh</div></div>
          <div className="stat-card ai"><div className="stat-top"><span>豆包修正累计</span><span className="stat-icon">✦</span></div><div className="stat-value">{fmt(data.metrics.aiTotal, 0)} <small>{data.metrics.aiTotal == null ? '' : 'kWh'}</small></div><div className="stat-foot">{aiReady ? `${data.metrics.aiEvaluatedDays} 天已调用 · MAE ${fmt(data.metrics.aiMae)} kWh` : '待配置 Ark API 密钥并运行真实调用'}</div></div>
        </div>
        {aiReady && <div className="result-note"><span className="notice-icon">i</span><span>本次回测中，豆包修正的日均绝对误差为 {fmt(data.metrics.aiMae, 2)} kWh，物理基线为 {fmt(data.metrics.physicsMae, 2)} kWh。{(data.metrics.aiMae ?? 0) >= (data.metrics.physicsMae ?? 0) ? '当前修正尚未带来精度提升。' : '本次修正降低了误差，仍需跨月份验证。'}</span></div>}
        <section className="panel trend-panel" id="daily"><div className="panel-head"><div><span className="section-kicker">DAILY FORECAST</span><h2>每日发电量对比</h2><p>点击曲线上的日期，查看当日天气预报和预测详情。</p></div><div className="small-badge">单位 kWh</div></div><Chart setSelected={setSelected} /></section>
        {day && <div className="detail-grid"><section className="panel day-panel"><div className="panel-head compact"><div><span className="section-kicker">DAY INSIGHT</span><h2>{day.date.slice(5).replace('-', ' 月 ')} 日 <span className="normal-title">预测拆解</span></h2></div><select value={selected} onChange={e => setSelected(e.target.value)} aria-label="选择日期">{data.daily.map(d => <option key={d.date} value={d.date}>{d.date}</option>)}</select></div><div className="day-metrics"><div><span>实际</span><strong>{fmt(day.actual)}</strong><small>kWh</small></div><div><span>物理预测</span><strong>{fmt(day.physics)}</strong><small>kWh</small></div><div><span>豆包修正</span><strong>{fmt(day.ai)}</strong><small>{day.ai == null ? '待运行' : 'kWh'}</small></div></div><div className="day-note">物理预测偏差 <b>{day.physics >= day.actual ? '+' : ''}{fmt(day.physics - day.actual)} kWh</b> · 近 30 天滚动系数 {day.adaptiveFactor.toFixed(3)} · GFS 运行时刻 {day.runUtc.slice(0, 10)} 00:00 UTC</div>{day.aiReason && <div className="ai-reason">豆包修正系数 {day.aiFactor?.toFixed(3)} · {day.aiReason}</div>}</section><section className="panel weather-panel"><div className="panel-head compact"><div><span className="section-kicker">WEATHER INPUT</span><h2>预报辐照度</h2></div><span className="weather-unit">W/m²</span></div><WeatherChart day={day} /><p>GFS 预报短波辐射 · 站点附近 0.25° 网格 · 当地时间</p></section></div>}
        <section className="future-section" id="future">
          <div className="section-heading"><div><span className="section-kicker">OUT-OF-DATASET FORECAST</span><h2>无实测新预测 · 2024 年 1 月</h2></div><span className="muted">HKUST SQ1 · 31 天</span></div>
          <div className="future-notice"><span className="notice-icon">i</span><span>公开数据集的发电实绩止于 {future.meta.lastActualDate}。2024 年 1 月没有真实发电量；这里使用前一天发布、预报目标日天气的 GFS 数据，并按发布时间重放。它属于数据集外的离线预测样例，不是当前实时预报，也无法计算预测误差。</span></div>
          <div className="future-stat-grid">
            <div className="stat-card physics"><div className="stat-top"><span>物理预测月累计</span><span className="stat-icon">☼</span></div><div className="stat-value">{fmt(future.metrics.physicsTotal, 0)} <small>kWh</small></div><div className="stat-foot">逐日预报结果之和 · 站点系数冻结于 {future.meta.calibrationFrozenThrough}</div></div>
            <div className="stat-card ai"><div className="stat-top"><span>豆包修正月累计</span><span className="stat-icon">✦</span></div><div className="stat-value">{fmt(future.metrics.aiTotal, 0)} <small>{future.metrics.aiTotal == null ? '' : 'kWh'}</small></div><div className="stat-foot">{future.meta.aiStatus === 'ready' ? '31 天真实 Ark API 输出 · 无实绩评分' : '豆包结果待完成 · 不显示模拟值'}</div></div>
            <div className="stat-card no-actual"><div className="stat-top"><span>实际发电量 / 误差</span><span className="stat-icon">—</span></div><div className="stat-value">无数据</div><div className="stat-foot">没有该月站点实绩，因此 MAE、MAPE 不适用</div></div>
          </div>
          <section className="panel future-chart-panel"><div className="panel-head"><div><span className="section-kicker">DAILY OUTLOOK</span><h2>每日预测曲线</h2><p>仅展示物理与豆包预测，未绘制任何“实际发电量”。</p></div><span className="small-badge">单位 kWh</span></div><FutureChart /></section>
        </section>
        <section className="panel split-panel"><div className="panel-head"><div><span className="section-kicker">DATA SPLIT & LABELS</span><h2>数据划分与使用边界</h2><p>“训练集”在此指物理参数校准；豆包没有用 SQ1 数据微调。2023 年实测并未丢弃，而是按预测时点使用。</p></div></div><div className="split-table"><div className="split-row header"><span>用途</span><span>时间 / 数量</span><span>数据及作用</span></div><div className="split-row"><strong>物理参数校准</strong><span>2021-06—2022-12 · 572 有效日</span><span>HKUST SQ1 发电量与附近气象观测，拟合固定站点系数</span></div><div className="split-row"><strong>2023 年实测</strong><span>截至各预测时点已完成的日期</span><span>用于近 30 天滚动校正和豆包历史误差参考；目标日实绩不进入预测输入</span></div><div className="split-row"><strong>历史回测</strong><span>2023-08 · 31 日</span><span>前一日发布的 GFS 对目标日天气的预报驱动预测；目标日实绩仅在事后评分</span></div><div className="split-row"><strong>无实测预测</strong><span>2024-01 · 31 日</span><span>逐日 GFS 归档预报；物理站点校正冻结于 2023-12-30；无真实发电量</span></div><div className="split-row"><strong>豆包大模型</strong><span>Ark API · 未微调</span><span>读取当时可用的物理预测与历史误差，输出受限修正系数并保存到本地</span></div></div></section>
        <section className="method-section" id="method"><div className="section-heading"><div><span className="section-kicker">METHODOLOGY</span><h2>预测是怎样得到的</h2></div><span className="muted">物理模型 × 大模型修正</span></div><div className="method-grid">{data.method.map(item => <article className="method-card" key={item.step}><span className="method-num">{item.step}</span><h3>{item.title}</h3><p>{item.text}</p></article>)}</div><div className="method-foot"><b>物理模型说明</b><span>采用 pvlib 的简化 PVWatts 直流功率公式与 Ross 电池温度近似。因公开数据未给出 SQ1 可核实的组件倾角、方位角，历史拟合系数不能解释为纯系统损耗。8 月回测逐日使用预报发布时可用的近 30 天资料校正；2024 年 1 月因无站点新实绩，将校正系数冻结在 2023-12-30。两个月度卡片都是逐日预测之和。</span></div></section>
        <section className="panel source-panel" id="sources"><div className="panel-head"><div><span className="section-kicker">DATA PROVENANCE</span><h2>数据与方法来源</h2><p>每一项输入都标明来源和用途，便于复现。</p></div><span className="small-badge">{data.sources.length} 项来源</span></div><div className="source-list">{data.sources.map((source, i) => <a href={source.url} target="_blank" rel="noreferrer" className="source-row" key={source.name}><span className="source-index">0{i + 1}</span><span className="source-main"><strong>{source.name}</strong><small>{source.use}</small></span><span className="source-provider">{source.provider}</span><span className="external">↗</span></a>)}</div></section>
        <footer><span>SolarScope · 光伏预测研究演示</span><span>公开数据回测 · 豆包模型：{data.meta.aiModel}</span></footer>
      </div>
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
