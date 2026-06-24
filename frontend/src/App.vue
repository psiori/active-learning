<template>
  <Transition name="startup-swap" mode="out-in">
    <BackendWaitingPlaceholder
      v-if="showBackendPlaceholder"
      key="backend-waiting"
      :status-text="backendStatusText"
    />

    <AppShell v-else key="app-shell">
      <template #sidebar>
        <section class="stage-card">
          <div class="stage-header">
            <p class="stage-kicker">CRID</p>
            <h1>Active Learning</h1>
          </div>
          <ProjectSelector
            :projects="projects"
            :selected-project="selectedProject"
            :loading-config="loadingConfig"
            @change="setSelectedProject"
          />
          <ProjectSummary
            :project-config="projectConfig"
            :seeded-ids-count="activeResult?.seeded_ids_count ?? null"
          />
          <button
            v-if="selectedProject"
            class="secondary-button"
            :disabled="loadingQuery || loadingStrategy"
            @click="startFresh"
          >
            Start fresh
          </button>
        </section>

        <QueryStageCard
          :query-form="queryForm"
          :loading-query="loadingQuery"
          :disabled="loadingQuery || loadingStrategy || !selectedProject"
          :cancelling-query="cancellingQuery"
          :is-dirty="isQueryDirty"
          @submit="runQueryStage"
          @cancel="cancelQueryStage"
        />

        <StrategyStageCard
          :selection-form="selectionForm"
          :strategy-params="strategyParams"
          :available-strategies="availableStrategies"
          :feature-models="featureModels"
          :uncertainty-models="uncertaintyModels"
          :uses-coreset-params="usesCoresetParams"
          :uses-uncertainty-params="usesUncertaintyParams"
          :uses-alges-params="usesAlgesParams"
          :selected-uncertainty-model="selectedUncertaintyModel"
          :selected-alges-model="selectedAlgesModel"
          :loading-strategy="loadingStrategy"
          :disabled-select="loadingQuery || loadingStrategy || !availableStrategies.length"
          :disabled-submit="loadingQuery || loadingStrategy || !selectedProject"
          :cancelling-strategy="cancellingStrategy"
          :is-dirty="isStrategyDirty"
          @submit="runStrategyStage"
          @cancel="cancelStrategyStage"
          @sync-model-urls="syncStrategyModelUrls"
        />

        <ExportStageCard
          :export-form="exportForm"
          :active-result="activeResult"
          :selected-strategy-ids="selectedStrategyIds"
          :sama-project-id="projectConfig?.project?.sama_project_id ?? null"
          :export-prefix="projectConfig?.project?.export_prefix || ''"
          :loading-export-overlay="loadingExportOverlay"
          :overlay-export-completed-at="overlayExportCompletedAt"
          :seed-confirmation="seedConfirmation"
          :loading-export-seed="loadingExportSeed"
          :disabled-input="loadingQuery || loadingStrategy || loadingExportOverlay || loadingExportSeed || !selectedProject"
          :disable-overlay="loadingQuery || loadingStrategy || loadingExportOverlay || !canDownloadOverlay"
          :disable-seed="loadingQuery || loadingStrategy || loadingExportSeed || !canSeedSelection"
          :is-dirty="isExportDirty"
          @overlay="runOverlayExport"
          @seed="handleSeedClick"
        />
      </template>

      <template #main>
        <PreviewPane
          :result-title="resultTitle"
          :error-message="errorMessage"
          :export-result="exportResult"
          :active-result="activeResult"
          :selected-strategy-ids="selectedStrategyIds"
          :show-job-progress="showJobProgress"
          :active-job="activeJob"
          :active-job-percent="activeJobPercent"
          :active-job-stages="activeJobStages"
          :show-overlay-controls="showOverlayControls"
          :overlay-alpha="overlayAlpha"
          :show-preview-grid="showPreviewGrid"
          :preview-items="previewItems"
          :overlay-opacity="overlayOpacity"
          :is-deselected="isStrategyItemDeselected"
          :show-selection-state="activeResult?.result_kind === 'strategy'"
          :loading-export-tags="loadingExportTags"
          :loading-export-overlay="loadingExportOverlay"
          :loading-export-seed="loadingExportSeed"
          :exclusion-tags-persisted-at="exclusionTagsPersistedAt"
          :seed-confirmation="seedConfirmation"
          :excluded-count="excludedStrategyIds.length"
          :total-count="activeResult?.selected_ids?.length || 0"
          :with-cache-root="withCacheRoot"
          :empty-state-text="emptyStateText"
          @toggle="togglePreviewSelection"
          @open="openLightbox"
          @update:overlay-alpha="overlayAlpha = $event"
          @export-tags="runExportExclusionTags"
          @include-all="includeAllStrategyItems"
          @exclude-all="excludeAllStrategyItems"
          @download-mosaic="runOverlayExport"
          @confirm-seed="confirmSeedExport"
        />

        <PreviewLightbox
          v-if="lightboxItem"
          :item="lightboxItem"
          :previous-item="lightboxPreviousItem"
          :next-item="lightboxNextItem"
          :overlay-alpha="overlayAlpha"
          :overlay-opacity="overlayOpacity"
          :show-overlay-controls="showOverlayControls"
          :show-selection-state="activeResult?.result_kind === 'strategy'"
          :is-excluded="lightboxItem ? isStrategyItemDeselected(lightboxItem.sample_id) : false"
          :with-cache-root="withCacheRoot"
          :has-multiple-items="previewItems.length > 1"
          @close="closeLightbox"
          @next="showNextLightboxItem"
          @previous="showPreviousLightboxItem"
          @update:overlay-alpha="overlayAlpha = $event"
          @toggle="lightboxItem && togglePreviewSelection(lightboxItem.sample_id)"
        />
      </template>
    </AppShell>
  </Transition>
</template>

<script setup>
import { computed, onMounted, onUnmounted } from 'vue'
import AppShell from './components/layout/AppShell.vue'
import BackendWaitingPlaceholder from './components/layout/BackendWaitingPlaceholder.vue'
import ExportStageCard from './components/export/ExportStageCard.vue'
import ProjectSelector from './components/project/ProjectSelector.vue'
import ProjectSummary from './components/project/ProjectSummary.vue'
import QueryStageCard from './components/query/QueryStageCard.vue'
import PreviewLightbox from './components/results/PreviewLightbox.vue'
import PreviewPane from './components/results/PreviewPane.vue'
import StrategyStageCard from './components/strategy/StrategyStageCard.vue'
import { usePreviewLightbox } from './composables/usePreviewLightbox'
import { useActiveLearningStore } from './stores/activeLearningStore'
import { sampleIdSortKey } from './utils/sampleId'

const store = useActiveLearningStore()

const projects = computed(() => store.projects)
const selectedProject = computed(() => store.selectedProject)
const projectConfig = computed(() => store.projectConfig)
const queryForm = computed(() => store.queryForm)
const selectionForm = computed(() => store.selectionForm)
const strategyParams = computed(() => store.strategyParams)
const exportForm = computed(() => store.exportForm)
const activeResult = computed(() => store.activeResult)
const loadingConfig = computed(() => store.loadingConfig)
const loadingQuery = computed(() => store.loadingQuery)
const loadingStrategy = computed(() => store.loadingStrategy)
const cancellingQuery = computed(() => store.cancellingQuery)
const cancellingStrategy = computed(() => store.cancellingStrategy)
const loadingExportOverlay = computed(() => store.loadingExportOverlay)
const overlayExportCompletedAt = computed(() => store.overlayExportCompletedAt)
const loadingExportSeed = computed(() => store.loadingExportSeed)
const errorMessage = computed(() => store.errorMessage)
const exportResult = computed(() => store.exportResult)
const selectedStrategyIds = computed(() => store.selectedStrategyIds)
const activeJob = computed(() => store.activeJob)
const usesCoresetParams = computed(() => store.usesCoresetParams)
const usesUncertaintyParams = computed(() => store.usesUncertaintyParams)
const usesAlgesParams = computed(() => store.usesAlgesParams)
const selectedUncertaintyModel = computed(() => store.selectedUncertaintyModel)
const selectedAlgesModel = computed(() => store.selectedAlgesModel)
const resultTitle = computed(() => store.resultTitle)
const emptyStateText = computed(() => store.emptyStateText)
const activeJobStages = computed(() => store.activeJobStages)
const activeJobPercent = computed(() => store.activeJobPercent)
const showJobProgress = computed(() => store.showJobProgress)
const overlayOpacity = computed(() => store.overlayOpacity)
const showOverlayControls = computed(() => store.showOverlayControls)
const canDownloadOverlay = computed(() => store.canDownloadOverlay)
const canSeedSelection = computed(() => store.canSeedSelection)
const showPreviewGrid = computed(() => store.showPreviewGrid)
const previewItems = computed(() => {
  const items = store.activeResult?.preview_items || []
  return [...items].sort(
    (a, b) => sampleIdSortKey(a.sample_id) - sampleIdSortKey(b.sample_id),
  )
})
const overlayAlpha = computed({
  get: () => store.overlayAlpha,
  set: (value) => {
    store.overlayAlpha = value
  },
})
const availableStrategies = computed(() => store.projectConfig?.selection?.available_strategies || [])
const featureModels = computed(() => store.modelOptionsForType('resnet50'))
const uncertaintyModels = computed(() => store.modelOptionsForType('unet'))

const loadingExportTags = computed(() => store.loadingExportTags)
const exclusionTagsPersistedAt = computed(() => store.exclusionTagsPersistedAt)
const seedConfirmation = computed(() => store.seedConfirmation)
const isQueryDirty = computed(() => store.isQueryDirty)
const isStrategyDirty = computed(() => store.isStrategyDirty)
const isExportDirty = computed(() => store.isExportDirty)
const excludedStrategyIds = computed(() => store.excludedStrategyIds)
const showBackendPlaceholder = computed(() => !store.backendReachable)
const backendStatusText = computed(() => store.backendStatusText)

const {
  bootstrap,
  closeJobStream,
  confirmSeedExport,
  excludeAllStrategyItems,
  includeAllStrategyItems,
  isStrategyItemDeselected,
  runExportExclusionTags,
  runOverlayExport,
  runQueryStage,
  runStrategyStage,
  cancelQueryStage,
  cancelStrategyStage,
  setSelectedProject,
  stageSeedConfirmation,
  startFresh,
  stopBootstrapRetry,
  syncStrategyModelUrls,
  togglePreviewSelection,
  withCacheRoot,
} = store

function handleSeedClick() {
  if (seedConfirmation.value) {
    confirmSeedExport()
    return
  }
  stageSeedConfirmation()
}

const {
  lightboxItem,
  lightboxPreviousItem,
  lightboxNextItem,
  openLightbox,
  closeLightbox,
  showPreviousLightboxItem,
  showNextLightboxItem,
} = usePreviewLightbox(previewItems)

onMounted(async () => {
  await bootstrap()
})

onUnmounted(() => {
  stopBootstrapRetry()
  closeJobStream()
})
</script>
