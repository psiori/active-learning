import { computed, nextTick, reactive, ref, watch } from 'vue'
import { StorageSerializers, useStorage } from '@vueuse/core'
import { defineStore } from 'pinia'
import {
  cancelJob,
  createJob,
  downloadOverlayMosaic,
  exportExclusionTags,
  fetchConfig,
  fetchJob,
  fetchProject,
  patchProjectConfig,
  seedStrategySelection,
} from '../api/activeLearningApi'
import {
  JOB_STAGE_LABELS,
  QUERY_STAGE_IDS,
  SELECTED_PROJECT_STORAGE_KEY,
  SESSION_STORAGE_KEY,
  STRATEGY_STAGE_IDS,
} from '../constants/activeLearning'

export const useActiveLearningStore = defineStore('activeLearning', () => {
  const BOOTSTRAP_RETRY_DELAY_MS = 2000
  const persistedSession = useStorage(SESSION_STORAGE_KEY, null, undefined, { serializer: StorageSerializers.object })
  let _suppressSessionWrites = true
  let _pendingAutoRerun = false
  const projects = ref([])
  const selectedProject = ref('')
  const projectConfig = ref(null)
  const activeResult = ref(null)
  const loadingConfig = ref(false)
  const loadingQuery = ref(false)
  const loadingStrategy = ref(false)
  const cancellingQuery = ref(false)
  const cancellingStrategy = ref(false)
  const loadingExportOverlay = ref(false)
  const loadingExportSeed = ref(false)
  const loadingExportTags = ref(false)
  const exclusionTagsPersistedAt = ref(0)
  const overlayExportCompletedAt = ref(0)
  const errorMessage = ref('')
  const backendReachable = ref(false)
  const waitingForBackend = ref(false)
  const backendRetryCount = ref(0)
  const cacheRoot = ref('data/active_learning/feature_cache')
  const overlayAlpha = ref(45)
  const queryResultToken = ref('')
  const overlayAvailable = ref(false)
  const exportResult = ref(null)
  const seedConfirmation = ref(null)
  const selectedStrategyIds = ref([])
  const activeJobId = ref('')
  const activeJob = ref(null)
  let currentEventSource = null

  const queryBaseline = ref(null)
  const strategyBaseline = ref(null)
  const exportBaseline = ref(null)

  /** Brightness sliders require finite numbers; API/session may send null. */
  function coerceBrightness(value, fallback) {
    const n = typeof value === 'number' ? value : Number(value)
    return Number.isFinite(n) ? n : fallback
  }

  const queryForm = reactive({
    exclude_seeded: false,
    exclude_al_excluded: false,
    brightness_filter_enabled: true,
    min_brightness: 0,
    max_brightness: 255,
    start: '',
    end: '',
    use_full_res_images: false,
    min_milliseconds_between_images: 0,
  })

  function normalizeQueryFormBrightness() {
    queryForm.min_brightness = coerceBrightness(queryForm.min_brightness, 0)
    queryForm.max_brightness = coerceBrightness(queryForm.max_brightness, 255)
  }

  const selectionForm = reactive({
    strategy: '',
    n_select: 50,
  })

  const strategyParams = reactive({
    feature_model: '',
    uncertainty_model: '',
    uncertainty_model_url: '',
    alpha: 0.5,
    provider: 'mc_dropout',
    mc_iterations: 5,
    batch_size: 32,
    aggregation: 'topk_mean',
    topk_fraction: 0.1,
    candidate_multiplier: 4,
    alges_model: '',
    alges_model_url: '',
    method: 'semantic',
    alges_batch_size: 32,
  })

  const exportForm = reactive({
    sama_priority: 0,
  })

  function modelOptionsForType(type) {
    return (projectConfig.value?.models || []).filter((model) => model.type === type)
  }

  function selectedModelUrl(modelName) {
    const match = (projectConfig.value?.models || []).find((model) => model.name === modelName)
    return match?.url || ''
  }

  const usesCoresetParams = computed(() => (
    ['coreset', 'uncertainty_coreset', 'uncertainty_topk_coreset', 'alges_coreset'].includes(selectionForm.strategy)
  ))

  const usesUncertaintyParams = computed(() => (
    ['uncertainty_coreset', 'uncertainty_topk', 'uncertainty_topk_coreset'].includes(selectionForm.strategy)
  ))

  const usesAlgesParams = computed(() => (
    ['alges', 'alges_coreset'].includes(selectionForm.strategy)
  ))

  const selectedUncertaintyModel = computed(() => (
    (projectConfig.value?.models || []).find((model) => model.name === strategyParams.uncertainty_model) || null
  ))

  const selectedAlgesModel = computed(() => (
    (projectConfig.value?.models || []).find((model) => model.name === strategyParams.alges_model) || null
  ))

  const resultTitle = computed(() => {
    if (exportResult.value) {
      return 'Export Results'
    }
    if (seedConfirmation.value) {
      return 'Seed'
    }
    if (activeJob.value?.state === 'running' || activeJob.value?.state === 'queued') {
      return activeJob.value.kind === 'strategy' ? 'Running Strategy' : 'Running Query'
    }
    if (activeJob.value?.state === 'failed') {
      return 'Job Failed'
    }
    if (activeResult.value?.result_kind === 'query') {
      return 'Query Results'
    }
    if (activeResult.value?.result_kind === 'strategy') {
      return 'Strategy Results'
    }
    return 'Preview'
  })

  const emptyStateText = computed(() => {
    if (loadingQuery.value) {
      return 'Loading query preview thumbnails…'
    }
    if (loadingStrategy.value) {
      return 'Running strategy…'
    }
    if (loadingExportOverlay.value) {
      return 'Preparing overlay mosaic download…'
    }
    if (loadingExportSeed.value) {
      return 'Exporting selected IDs to CRID and Sama…'
    }
    return 'Run query or strategy to see a thumbnail sample.'
  })

  const activeJobStages = computed(() => {
    const kind = activeJob.value?.kind
    if (!kind) {
      return []
    }
    const stageIds = kind === 'strategy' ? STRATEGY_STAGE_IDS : QUERY_STAGE_IDS
    const currentIndex = stageIds.indexOf(activeJob.value?.stage)

    const skipped = new Set(activeJob.value?.skipped_stages || [])
    const cached = new Set(activeJob.value?.cached_stages || [])

    return stageIds.map((id, index) => {
      let state = 'pending'

      if (cached.has(id)) {
        state = 'cached'
      } else if (skipped.has(id)) {
        state = 'skipped'
      } else if (activeJob.value?.state === 'failed' && id === activeJob.value?.stage) {
        state = 'failed'
      } else if (index < currentIndex || activeJob.value?.state === 'completed') {
        state = 'completed'
      } else if (id === activeJob.value?.stage) {
        state = activeJob.value?.state === 'completed' ? 'completed' : 'running'
      }

      return {
        id,
        label: JOB_STAGE_LABELS[id],
        state,
      }
    })
  })

  const activeJobPercent = computed(() => activeJob.value?.percent ?? 0)

  const showJobProgress = computed(() => (
    Boolean(!exportResult.value && activeJob.value && ['queued', 'running', 'failed'].includes(activeJob.value.state))
  ))

  const overlayOpacity = computed(() => overlayAlpha.value / 100)

  const showOverlayControls = computed(() => Boolean(
    !exportResult.value
    && activeResult.value?.result_kind === 'strategy'
    && activeResult.value?.preview_items?.some((item) => item.mask_url),
  ))

  const canDownloadOverlay = computed(() => (
    activeResult.value?.result_kind === 'strategy'
    && overlayAvailable.value
    && selectedStrategyIds.value.length > 0
  ))

  const canSeedSelection = computed(() => (
    activeResult.value?.result_kind === 'strategy' && selectedStrategyIds.value.length > 0
  ))

  const excludedStrategyIds = computed(() => {
    if (activeResult.value?.result_kind !== 'strategy') return []
    const selected = new Set(selectedStrategyIds.value)
    return (activeResult.value.selected_ids || []).filter((id) => !selected.has(id))
  })

  const showPreviewGrid = computed(() => (
    !exportResult.value && Boolean(activeResult.value?.preview_items?.length)
  ))

  function snapshotQuery() {
    return {
      exclude_seeded: queryForm.exclude_seeded,
      exclude_al_excluded: queryForm.exclude_al_excluded,
      brightness_filter_enabled: queryForm.brightness_filter_enabled,
      min_brightness: queryForm.min_brightness,
      max_brightness: queryForm.max_brightness,
      start: queryForm.start,
      end: queryForm.end,
      use_full_res_images: queryForm.use_full_res_images,
      min_milliseconds_between_images: queryForm.min_milliseconds_between_images,
    }
  }

  function snapshotStrategy() {
    return {
      strategy: selectionForm.strategy,
      n_select: selectionForm.n_select,
      feature_model: strategyParams.feature_model,
      uncertainty_model: strategyParams.uncertainty_model,
      uncertainty_model_url: strategyParams.uncertainty_model_url,
      alpha: strategyParams.alpha,
      provider: strategyParams.provider,
      mc_iterations: strategyParams.mc_iterations,
      batch_size: strategyParams.batch_size,
      aggregation: strategyParams.aggregation,
      topk_fraction: strategyParams.topk_fraction,
      candidate_multiplier: strategyParams.candidate_multiplier,
      alges_model: strategyParams.alges_model,
      alges_model_url: strategyParams.alges_model_url,
      method: strategyParams.method,
      alges_batch_size: strategyParams.alges_batch_size,
    }
  }

  function snapshotExport() {
    return {
      sama_priority: exportForm.sama_priority,
    }
  }

  function writePersistedSession(visibleResultKind) {
    const strat = snapshotStrategy()
    persistedSession.value = {
      version: 1,
      project_name: selectedProject.value,
      visible_result_kind: visibleResultKind,
      forms: {
        query: snapshotQuery(),
        selection: { strategy: strat.strategy, n_select: strat.n_select },
        strategy_params: {
          feature_model: strat.feature_model,
          uncertainty_model: strat.uncertainty_model,
          uncertainty_model_url: strat.uncertainty_model_url,
          alpha: strat.alpha,
          provider: strat.provider,
          mc_iterations: strat.mc_iterations,
          batch_size: strat.batch_size,
          aggregation: strat.aggregation,
          topk_fraction: strat.topk_fraction,
          candidate_multiplier: strat.candidate_multiplier,
          alges_model: strat.alges_model,
          alges_model_url: strat.alges_model_url,
          method: strat.method,
          alges_batch_size: strat.alges_batch_size,
        },
        export: snapshotExport(),
      },
      ui: { overlay_alpha: overlayAlpha.value },
      strategy_view: visibleResultKind === 'strategy'
        ? {
            active_result: activeResult.value,
            selected_strategy_ids: [...selectedStrategyIds.value],
          }
        : null,
    }
  }

  function clearPersistedSession() {
    persistedSession.value = null
  }

  function hydrateFromSession() {
    const session = persistedSession.value
    if (!session || typeof session !== 'object' || session.version !== 1) {
      if (session) clearPersistedSession()
      return
    }
    if (session.project_name !== selectedProject.value) {
      return
    }

    if (session.forms?.query) {
      Object.assign(queryForm, session.forms.query)
      normalizeQueryFormBrightness()
    }
    if (session.forms?.selection) {
      selectionForm.strategy = session.forms.selection.strategy
      selectionForm.n_select = session.forms.selection.n_select
    }
    if (session.forms?.strategy_params) {
      Object.assign(strategyParams, session.forms.strategy_params)
    }
    if (session.forms?.export) {
      Object.assign(exportForm, session.forms.export)
    }
    if (session.ui?.overlay_alpha != null) {
      overlayAlpha.value = session.ui.overlay_alpha
    }

    if (session.visible_result_kind === 'strategy' && session.strategy_view?.active_result) {
      activeResult.value = session.strategy_view.active_result
      selectedStrategyIds.value = session.strategy_view.selected_strategy_ids || []
      overlayAvailable.value = Boolean(activeResult.value?.overlay_available)
    } else if (session.visible_result_kind === 'query') {
      _pendingAutoRerun = true
    }
  }

  watch(
    [queryForm, selectionForm, strategyParams, exportForm, overlayAlpha, activeResult, selectedStrategyIds],
    () => {
      if (!_suppressSessionWrites) {
        writePersistedSession(activeResult.value?.result_kind || 'none')
      }
    },
    { deep: true },
  )

  const isQueryDirty = computed(() => {
    if (!queryBaseline.value) return false
    return JSON.stringify(snapshotQuery()) !== JSON.stringify(queryBaseline.value)
  })

  const isStrategyDirty = computed(() => {
    if (!strategyBaseline.value) return false
    return JSON.stringify(snapshotStrategy()) !== JSON.stringify(strategyBaseline.value)
  })

  const isExportDirty = computed(() => {
    if (!exportBaseline.value) return false
    return JSON.stringify(snapshotExport()) !== JSON.stringify(exportBaseline.value)
  })

  const backendStatusText = computed(() => {
    if (backendReachable.value) {
      return ''
    }

    return 'Connecting to backend..'
  })

  let bootstrapRetryTimer = null

  function isBackendUnreachable(error) {
    return !error?.response
  }

  function clearBootstrapRetry() {
    if (bootstrapRetryTimer) {
      window.clearTimeout(bootstrapRetryTimer)
      bootstrapRetryTimer = null
    }
  }

  function scheduleBootstrapRetry() {
    clearBootstrapRetry()
    bootstrapRetryTimer = window.setTimeout(() => {
      bootstrap()
    }, BOOTSTRAP_RETRY_DELAY_MS)
  }

  function syncQueryForm(query) {
    if (query.cache_root) {
      cacheRoot.value = query.cache_root
    }
    exportForm.sama_priority = query.sama_priority ?? 0
    queryForm.exclude_seeded = query.exclude_seeded
    queryForm.exclude_al_excluded = query.exclude_al_excluded ?? false
    queryForm.brightness_filter_enabled = query.brightness_filter_enabled !== false
    queryForm.min_brightness = coerceBrightness(query.min_brightness, 0)
    queryForm.max_brightness = coerceBrightness(query.max_brightness, 255)
    queryForm.start = query.start.slice(0, 10)
    queryForm.end = query.end.slice(0, 10)
    queryForm.use_full_res_images = query.use_full_res_images
    queryForm.min_milliseconds_between_images = query.min_milliseconds_between_images
  }

  function resetProjectScopedState() {
    errorMessage.value = ''
    activeResult.value = null
    exportResult.value = null
    seedConfirmation.value = null
    exclusionTagsPersistedAt.value = 0
    queryResultToken.value = ''
    overlayAvailable.value = false
    selectedStrategyIds.value = []
    activeJobId.value = ''
    activeJob.value = null
    closeJobStream()
  }

  async function loadProject(projectName) {
    if (!projectName) {
      return
    }

    projectConfig.value = await fetchProject(projectName)
    syncQueryForm(projectConfig.value.query)
    selectionForm.strategy = projectConfig.value.selection.strategy
    selectionForm.n_select = projectConfig.value.selection.n_select
    strategyParams.feature_model = projectConfig.value.coreset.feature_model
    strategyParams.uncertainty_model = projectConfig.value.uncertainty_coreset.uncertainty_model
    strategyParams.uncertainty_model_url = selectedModelUrl(projectConfig.value.uncertainty_coreset.uncertainty_model)
    strategyParams.alpha = projectConfig.value.uncertainty_coreset.alpha
    strategyParams.provider = projectConfig.value.uncertainty_coreset.provider
    strategyParams.mc_iterations = projectConfig.value.uncertainty_coreset.mc_iterations
    strategyParams.batch_size = projectConfig.value.uncertainty_coreset.batch_size
    strategyParams.aggregation = projectConfig.value.uncertainty_coreset.aggregation
    strategyParams.topk_fraction = projectConfig.value.uncertainty_coreset.topk_fraction
    strategyParams.candidate_multiplier = projectConfig.value.uncertainty_coreset.candidate_multiplier
    strategyParams.alges_model = projectConfig.value.alges.model
    strategyParams.alges_model_url = selectedModelUrl(projectConfig.value.alges.model)
    strategyParams.method = projectConfig.value.alges.method
    strategyParams.alges_batch_size = projectConfig.value.alges.batch_size

    queryBaseline.value = snapshotQuery()
    strategyBaseline.value = snapshotStrategy()
    exportBaseline.value = snapshotExport()
  }

  async function bootstrap() {
    clearBootstrapRetry()
    loadingConfig.value = true
    errorMessage.value = ''

    try {
      const config = await fetchConfig()
      backendReachable.value = true
      waitingForBackend.value = false
      backendRetryCount.value = 0
      projects.value = config.projects

      if (projects.value.length > 0) {
        const storedProject = window.localStorage.getItem(SELECTED_PROJECT_STORAGE_KEY)
        const hasStoredProject = projects.value.some((project) => project.project_name === storedProject)
        selectedProject.value = hasStoredProject ? storedProject : projects.value[0].project_name
        await loadProject(selectedProject.value)
        hydrateFromSession()
        nextTick(() => {
          _suppressSessionWrites = false
          if (_pendingAutoRerun) {
            _pendingAutoRerun = false
            runQueryStage()
          }
        })
      }
    } catch (error) {
      if (isBackendUnreachable(error)) {
        backendReachable.value = false
        waitingForBackend.value = true
        backendRetryCount.value += 1
        scheduleBootstrapRetry()
      } else {
        waitingForBackend.value = false
        errorMessage.value = error.message
      }
    } finally {
      loadingConfig.value = false
    }
  }

  function buildJobPayload(kind) {
    const includeQueryToken = kind === 'strategy' && !isQueryDirty.value && Boolean(queryResultToken.value)
    return {
      project_name: selectedProject.value,
      query_result_token: includeQueryToken ? queryResultToken.value : null,
      strategy: selectionForm.strategy,
      n_select: selectionForm.n_select,
      min_milliseconds_between_images: queryForm.min_milliseconds_between_images,
      feature_model: strategyParams.feature_model,
      uncertainty_model: strategyParams.uncertainty_model,
      uncertainty_model_url: selectedUncertaintyModel.value?.url ? strategyParams.uncertainty_model_url : null,
      alpha: strategyParams.alpha,
      provider: strategyParams.provider,
      mc_iterations: strategyParams.mc_iterations,
      batch_size: strategyParams.batch_size,
      aggregation: strategyParams.aggregation,
      topk_fraction: strategyParams.topk_fraction,
      candidate_multiplier: strategyParams.candidate_multiplier,
      alges_model: strategyParams.alges_model,
      alges_model_url: selectedAlgesModel.value?.url ? strategyParams.alges_model_url : null,
      method: strategyParams.method,
      alges_batch_size: strategyParams.alges_batch_size,
      exclude_seeded: queryForm.exclude_seeded,
      exclude_al_excluded: queryForm.exclude_al_excluded,
      brightness_filter_enabled: queryForm.brightness_filter_enabled,
      min_brightness: queryForm.min_brightness,
      max_brightness: queryForm.max_brightness,
      start: queryForm.start,
      end: queryForm.end,
      use_full_res_images: queryForm.use_full_res_images,
      sample_size: 60,
    }
  }

  function buildStrategyExportContext() {
    if (activeResult.value?.result_kind !== 'strategy') {
      return null
    }

    return {
      project: { ...activeResult.value.project },
      query: { ...activeResult.value.query },
      selection: { ...activeResult.value.selection },
      models: [...(activeResult.value.models || [])],
      uncertainty_coreset: { ...activeResult.value.uncertainty_coreset },
      alges: { ...activeResult.value.alges },
      export: {
        sama_project_id: activeResult.value.project?.sama_project_id ?? null,
        sama_priority: exportForm.sama_priority,
      },
    }
  }

  function buildStrategyExportPayload(selectedIds = selectedStrategyIds.value) {
    const exportContext = buildStrategyExportContext()
    if (!exportContext) {
      return null
    }
    return {
      export_context: exportContext,
      selected_ids: [...selectedIds],
    }
  }

  async function startJob(kind) {
    closeJobStream()
    errorMessage.value = ''
    exportResult.value = null
    seedConfirmation.value = null
    exclusionTagsPersistedAt.value = 0
    activeJob.value = null

    if (kind === 'query') {
      loadingQuery.value = true
      loadingStrategy.value = false
    } else {
      loadingStrategy.value = true
      loadingQuery.value = false
    }

    try {
      const job = await createJob(kind, buildJobPayload(kind))
      activeJobId.value = job.job_id
      activeJob.value = job
      openJobStream(job.job_id)
      activeJob.value = await fetchJob(job.job_id)
    } catch (error) {
      errorMessage.value = error.message
      loadingQuery.value = false
      loadingStrategy.value = false
    }
  }

  async function runQueryStage() {
    if (!selectedProject.value) {
      return
    }

    await startJob('query')
  }

  async function runStrategyStage() {
    if (!selectedProject.value) {
      return
    }

    await startJob('strategy')
  }

  async function cancelQueryStage() {
    cancellingQuery.value = true
    try {
      if (activeJobId.value) {
        await cancelJob(activeJobId.value)
      }
    } catch {
      // ignore — job may have already finished
    }
    closeJobStream()
    activeResult.value = null
    errorMessage.value = ''
    activeJob.value = null
    activeJobId.value = ''
    loadingQuery.value = false
    cancellingQuery.value = false
  }

  async function cancelStrategyStage() {
    cancellingStrategy.value = true
    try {
      if (activeJobId.value) {
        await cancelJob(activeJobId.value)
      }
    } catch {
      // ignore — job may have already finished
    }
    closeJobStream()
    activeResult.value = null
    errorMessage.value = ''
    activeJob.value = null
    activeJobId.value = ''
    loadingStrategy.value = false
    cancellingStrategy.value = false
  }

  function openJobStream(jobId) {
    currentEventSource = new EventSource(`/api/al/jobs/${encodeURIComponent(jobId)}/events`)
    currentEventSource.addEventListener('snapshot', handleJobEvent)
    currentEventSource.addEventListener('status', handleJobEvent)
    currentEventSource.addEventListener('progress', handleJobEvent)
    currentEventSource.addEventListener('result', handleJobEvent)
    currentEventSource.addEventListener('error', handleJobEvent)
    currentEventSource.addEventListener('end', () => {
      closeJobStream()
    })
    currentEventSource.onerror = async () => {
      if (!activeJobId.value) {
        return
      }

      try {
        activeJob.value = await fetchJob(activeJobId.value)
      } catch (error) {
        errorMessage.value = error.message
      }
    }
  }

  function handleJobEvent(event) {
    const data = JSON.parse(event.data)
    activeJob.value = data

    if (data.state === 'completed' && data.result) {
      activeResult.value = {
        ...data.result,
        seeded_ids_count: data.result.seeded_ids_count ?? activeResult.value?.seeded_ids_count,
      }
      if (data.result.query?.cache_root) {
        cacheRoot.value = data.result.query.cache_root
      }
      queryResultToken.value = data.result.query_result_token || ''

      if (data.result.result_kind === 'strategy') {
        overlayAvailable.value = Boolean(data.result.overlay_available)
        const esExcluded = new Set(
          (data.result.preview_items || [])
            .filter((item) => item.excluded)
            .map((item) => item.sample_id),
        )
        selectedStrategyIds.value = (data.result.selected_ids || []).filter(
          (id) => !esExcluded.has(id),
        )
      }

      loadingQuery.value = false
      loadingStrategy.value = false
    } else if (data.state === 'failed') {
      errorMessage.value = data.error?.detail || data.message || 'Job failed'
      loadingQuery.value = false
      loadingStrategy.value = false
    }
  }

  function closeJobStream() {
    if (currentEventSource) {
      currentEventSource.close()
      currentEventSource = null
    }
  }

  async function runOverlayExport() {
    const payload = buildStrategyExportPayload()
    if (!payload) {
      return
    }

    loadingExportOverlay.value = true
    errorMessage.value = ''

    try {
      const { url, filename } = await downloadOverlayMosaic(payload)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = filename
      anchor.click()
      window.URL.revokeObjectURL(url)
      overlayExportCompletedAt.value = Date.now()
    } catch (error) {
      errorMessage.value = error.message
    } finally {
      loadingExportOverlay.value = false
    }
  }

  function stageSeedConfirmation() {
    const payload = buildStrategyExportPayload()
    if (!payload) {
      return
    }
    seedConfirmation.value = {
      export_id: payload.export_context?.project?.export_prefix
        ? `export-${payload.export_context.project.export_prefix}`
        : 'export-pending',
      image_count: payload.selected_ids.length,
      priority: payload.export_context?.export?.sama_priority ?? 0,
      sama_project_id: payload.export_context?.export?.sama_project_id ?? null,
      selected_ids: [...payload.selected_ids],
      payload,
    }
  }

  async function confirmSeedExport() {
    const payload = seedConfirmation.value?.payload || buildStrategyExportPayload()
    if (!payload) {
      return
    }

    loadingExportSeed.value = true
    errorMessage.value = ''

    try {
      const result = await seedStrategySelection(payload)

      exportResult.value = {
        kind: 'seed',
        message: result.sama_batch_id
          ? 'Selection exported and submitted to Sama.'
          : 'Selection exported to CRID.',
        ...result,
      }
      seedConfirmation.value = null
    } catch (error) {
      errorMessage.value = error.message
    } finally {
      loadingExportSeed.value = false
    }
  }

  async function runExportExclusionTags() {
    if (activeResult.value?.result_kind !== 'strategy' || excludedStrategyIds.value.length === 0) {
      return
    }

    loadingExportTags.value = true
    errorMessage.value = ''

    try {
      const result = await exportExclusionTags({
        selected_ids: [...(activeResult.value.selected_ids || [])],
        excluded_ids: excludedStrategyIds.value,
        project_name: selectedProject.value,
      })
      if ((result?.updated_count ?? 0) > 0) {
        exclusionTagsPersistedAt.value = Date.now()
        queryResultToken.value = ''
      }
    } catch (error) {
      errorMessage.value = error.message
    } finally {
      loadingExportTags.value = false
    }
  }

  function withCacheRoot(url) {
    const separator = url.includes('?') ? '&' : '?'
    return `${url}${separator}cache_root=${encodeURIComponent(cacheRoot.value)}`
  }

  function isStrategyItemDeselected(sampleId) {
    return activeResult.value?.result_kind === 'strategy'
      && !selectedStrategyIds.value.includes(sampleId)
  }

  function togglePreviewSelection(sampleId) {
    if (activeResult.value?.result_kind !== 'strategy') {
      return
    }

    const next = new Set(selectedStrategyIds.value)
    if (next.has(sampleId)) {
      next.delete(sampleId)
    } else {
      next.add(sampleId)
    }
    selectedStrategyIds.value = [...next]
  }

  function includeAllStrategyItems() {
    if (activeResult.value?.result_kind !== 'strategy') {
      return
    }

    selectedStrategyIds.value = [...(activeResult.value.selected_ids || [])]
  }

  function excludeAllStrategyItems() {
    if (activeResult.value?.result_kind !== 'strategy') {
      return
    }

    selectedStrategyIds.value = []
  }

  async function setSelectedProject(projectName) {
    if (!projectName || projectName === selectedProject.value) {
      return
    }

    writePersistedSession(activeResult.value?.result_kind || 'none')

    selectedProject.value = projectName
    window.localStorage.setItem(SELECTED_PROJECT_STORAGE_KEY, projectName)
    resetProjectScopedState()
    _suppressSessionWrites = true

    try {
      await loadProject(projectName)
      hydrateFromSession()
      nextTick(() => {
        _suppressSessionWrites = false
        if (_pendingAutoRerun) {
          _pendingAutoRerun = false
          runQueryStage()
        }
      })
    } catch (error) {
      _suppressSessionWrites = false
      errorMessage.value = error.message
    }
  }

  async function startFresh() {
    clearPersistedSession()
    resetProjectScopedState()
    overlayAlpha.value = 45
    _pendingAutoRerun = false
    _suppressSessionWrites = true

    try {
      await loadProject(selectedProject.value)
      nextTick(() => { _suppressSessionWrites = false })
    } catch (error) {
      _suppressSessionWrites = false
      errorMessage.value = error.message
    }
  }

  async function saveQueryConfig() {
    await patchProjectConfig(selectedProject.value, {
      query: snapshotQuery(),
    })
    queryBaseline.value = snapshotQuery()
  }

  async function saveExportConfig() {
    await patchProjectConfig(selectedProject.value, {
      query: { sama_priority: exportForm.sama_priority },
    })
    if (projectConfig.value?.query) {
      projectConfig.value.query.sama_priority = exportForm.sama_priority
    }
    exportBaseline.value = snapshotExport()
  }

  function updateModelUrl(modelName, url) {
    if (!projectConfig.value?.models || !modelName || !url) {
      return
    }

    const model = projectConfig.value.models.find((entry) => entry.name === modelName)
    if (model) {
      model.url = url
    }
  }

  async function saveStrategyConfig() {
    const snap = snapshotStrategy()
    const models = {}
    if (snap.uncertainty_model && snap.uncertainty_model_url) {
      models[snap.uncertainty_model] = { url: snap.uncertainty_model_url }
    }
    if (
      snap.alges_model
      && snap.alges_model_url
      && !Object.prototype.hasOwnProperty.call(models, snap.alges_model)
    ) {
      models[snap.alges_model] = { url: snap.alges_model_url }
    }
    await patchProjectConfig(selectedProject.value, {
      selection: { strategy: snap.strategy, n_select: snap.n_select },
      coreset: { feature_model: snap.feature_model },
      uncertainty_coreset: {
        uncertainty_model: snap.uncertainty_model,
        alpha: snap.alpha,
        provider: snap.provider,
        mc_iterations: snap.mc_iterations,
        batch_size: snap.batch_size,
        aggregation: snap.aggregation,
        topk_fraction: snap.topk_fraction,
        candidate_multiplier: snap.candidate_multiplier,
      },
      alges: {
        model: snap.alges_model,
        method: snap.method,
        batch_size: snap.alges_batch_size,
      },
      models,
    })
    updateModelUrl(snap.uncertainty_model, snap.uncertainty_model_url)
    updateModelUrl(snap.alges_model, snap.alges_model_url)
    strategyBaseline.value = snap
  }

  function syncStrategyModelUrls() {
    strategyParams.uncertainty_model_url = selectedModelUrl(strategyParams.uncertainty_model)
    strategyParams.alges_model_url = selectedModelUrl(strategyParams.alges_model)
  }

  function stopBootstrapRetry() {
    clearBootstrapRetry()
  }

  watch(
    () => JSON.stringify(snapshotQuery()),
    (next, previous) => {
      if (previous !== undefined && next !== previous) {
        queryResultToken.value = ''
      }
    },
  )

  return {
    projects,
    selectedProject,
    projectConfig,
    activeResult,
    loadingConfig,
    loadingQuery,
    loadingStrategy,
    cancellingQuery,
    cancellingStrategy,
    loadingExportOverlay,
    loadingExportSeed,
    loadingExportTags,
    exclusionTagsPersistedAt,
    overlayExportCompletedAt,
    seedConfirmation,
    errorMessage,
    backendReachable,
    waitingForBackend,
    backendRetryCount,
    backendStatusText,
    cacheRoot,
    overlayAlpha,
    queryResultToken,
    overlayAvailable,
    exportResult,
    selectedStrategyIds,
    excludedStrategyIds,
    activeJobId,
    activeJob,
    queryForm,
    selectionForm,
    strategyParams,
    exportForm,
    usesCoresetParams,
    usesUncertaintyParams,
    usesAlgesParams,
    selectedUncertaintyModel,
    selectedAlgesModel,
    resultTitle,
    emptyStateText,
    activeJobStages,
    activeJobPercent,
    showJobProgress,
    overlayOpacity,
    showOverlayControls,
    canDownloadOverlay,
    canSeedSelection,
    showPreviewGrid,
    isQueryDirty,
    isStrategyDirty,
    isExportDirty,
    modelOptionsForType,
    selectedModelUrl,
    bootstrap,
    stopBootstrapRetry,
    loadProject,
    setSelectedProject,
    syncQueryForm,
    startJob,
    openJobStream,
    handleJobEvent,
    closeJobStream,
    runQueryStage,
    runStrategyStage,
    cancelQueryStage,
    cancelStrategyStage,
    runOverlayExport,
    stageSeedConfirmation,
    confirmSeedExport,
    runExportExclusionTags,
    saveQueryConfig,
    saveStrategyConfig,
    saveExportConfig,
    excludeAllStrategyItems,
    includeAllStrategyItems,
    resetProjectScopedState,
    buildJobPayload,
    buildStrategyExportPayload,
    withCacheRoot,
    isStrategyItemDeselected,
    togglePreviewSelection,
    syncStrategyModelUrls,
    startFresh,
  }
})
