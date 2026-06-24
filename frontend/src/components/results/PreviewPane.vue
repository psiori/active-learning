<template>
  <section>
    <ActiveJobBanner
      v-if="showJobProgress"
      :active-job="activeJob"
      :active-job-percent="activeJobPercent"
      :active-job-stages="activeJobStages"
    />

    <ResultsHeader :result-title="resultTitle" :error-message="errorMessage" />

    <div class="preview-pane-offset" :style="spacerStyle" aria-hidden="true"></div>

    <div ref="paneRoot">
      <section v-if="seedConfirmation" class="seed-confirmation-panel">
        <div class="seed-confirmation-panel__header">
          <div>
            <p class="stage-kicker">Seed</p>
            <h2>Confirm before sending to Sama</h2>
            <p class="seed-confirmation-panel__lede">
              Review the export metadata and the final selected images one more time before confirming the seed.
            </p>
          </div>
        </div>

        <div class="seed-confirmation-panel__grid">
          <div class="seed-meta">
            <span>Export ID</span>
            <strong>{{ seedConfirmation.export_id }}</strong>
          </div>
          <div class="seed-meta">
            <span>Images</span>
            <strong>{{ seedConfirmation.image_count }}</strong>
          </div>
          <div class="seed-meta">
            <span>Priority</span>
            <strong>{{ seedConfirmation.priority }}</strong>
          </div>
          <div class="seed-meta">
            <span>Sama project</span>
            <strong>{{ seedConfirmation.sama_project_id ?? 'n/a' }}</strong>
          </div>
        </div>

        <section class="seed-controls-panel">
          <div v-if="showOverlayControls" class="seed-controls-panel__overlay">
            <p class="results-controls__label">Mask overlay opacity</p>
            <RangeSlider
              :min="0"
              :max="100"
              :step="1"
              :min-value="0"
              :max-value="overlayAlpha"
              :single-value="true"
              :format-value="formatOpacityValue"
              min-aria-label="Minimum mask overlay opacity"
              max-aria-label="Mask overlay opacity"
              @update:max-value="emit('update:overlayAlpha', $event)"
            />
          </div>
          <div class="seed-controls-panel__actions">
            <button class="secondary-button" type="button" :disabled="loadingExportOverlay" @click="emit('downloadMosaic')">
              {{ loadingExportOverlay ? 'Preparing...' : 'Download mosaic' }}
            </button>
            <button class="primary-button" type="button" :disabled="loadingExportSeed" @click="emit('confirmSeed')">
              {{ loadingExportSeed ? 'Confirming...' : 'Confirm seed' }}
            </button>
          </div>
        </section>
      </section>

      <StatsPanel
        v-else
        :export-result="exportResult"
        :active-result="activeResult"
        :selected-strategy-ids="selectedStrategyIds"
      />

      <ResultsControls
        v-if="!seedConfirmation && (showOverlayControls || showSelectionState)"
        :overlay-alpha="overlayAlpha"
        :show-overlay-controls="showOverlayControls"
        :show-selection-state="showSelectionState"
        :loading-export-tags="loadingExportTags"
        :exclusion-tags-persisted-at="exclusionTagsPersistedAt"
        :excluded-count="excludedCount"
        :total-count="totalCount"
        @update:overlay-alpha="emit('update:overlayAlpha', $event)"
        @export-tags="emit('exportTags')"
        @include-all="emit('includeAll')"
        @exclude-all="emit('excludeAll')"
      />

      <PreviewGrid
        v-if="showPreviewGrid"
        :preview-items="visiblePreviewItems"
        :overlay-opacity="overlayOpacity"
        :is-deselected="isDeselected"
        :show-selection-state="showSelectionState"
        :with-cache-root="withCacheRoot"
        @open="emit('open', $event)"
        @toggle="emit('toggle', $event)"
      />

      <EmptyStatePanel
        v-if="!showPreviewGrid && !showJobProgress && !seedConfirmation"
        :export-result="exportResult"
        :empty-state-text="emptyStateText"
      />
    </div>
  </section>
</template>

<script setup>
import { computed, ref, toRef } from 'vue'
import { usePreviewPaneOffset } from '@/composables/usePreviewPaneOffset'
import ActiveJobBanner from './ActiveJobBanner.vue'
import EmptyStatePanel from './EmptyStatePanel.vue'
import RangeSlider from '@/components/base/RangeSlider.vue'
import ResultsControls from './ResultsControls.vue'
import PreviewGrid from './PreviewGrid.vue'
import ResultsHeader from './ResultsHeader.vue'
import StatsPanel from './StatsPanel.vue'

const props = defineProps({
  resultTitle: {
    type: String,
    required: true,
  },
  errorMessage: {
    type: String,
    default: '',
  },
  exportResult: {
    type: Object,
    default: null,
  },
  seedConfirmation: {
    type: Object,
    default: null,
  },
  activeResult: {
    type: Object,
    default: null,
  },
  selectedStrategyIds: {
    type: Array,
    required: true,
  },
  showJobProgress: {
    type: Boolean,
    required: true,
  },
  activeJob: {
    type: Object,
    default: null,
  },
  activeJobPercent: {
    type: Number,
    required: true,
  },
  activeJobStages: {
    type: Array,
    required: true,
  },
  showOverlayControls: {
    type: Boolean,
    required: true,
  },
  overlayAlpha: {
    type: Number,
    required: true,
  },
  showPreviewGrid: {
    type: Boolean,
    required: true,
  },
  previewItems: {
    type: Array,
    required: true,
  },
  overlayOpacity: {
    type: Number,
    required: true,
  },
  isDeselected: {
    type: Function,
    required: true,
  },
  showSelectionState: {
    type: Boolean,
    required: true,
  },
  loadingExportTags: {
    type: Boolean,
    required: true,
  },
  loadingExportOverlay: {
    type: Boolean,
    required: true,
  },
  loadingExportSeed: {
    type: Boolean,
    required: true,
  },
  exclusionTagsPersistedAt: {
    type: Number,
    required: true,
  },
  excludedCount: {
    type: Number,
    required: true,
  },
  totalCount: {
    type: Number,
    required: true,
  },
  withCacheRoot: {
    type: Function,
    required: true,
  },
  emptyStateText: {
    type: String,
    required: true,
  },
})

const emit = defineEmits(['open', 'toggle', 'update:overlayAlpha', 'exportTags', 'includeAll', 'excludeAll', 'downloadMosaic', 'confirmSeed'])
const paneRoot = ref(null)
const visiblePreviewItems = computed(() => (
  props.seedConfirmation
    ? props.previewItems.filter((item) => !props.isDeselected(item.sample_id))
    : props.previewItems
))

function formatOpacityValue(value) {
  return `${value}%`
}

const { spacerStyle } = usePreviewPaneOffset({
  elementRef: paneRoot,
  showPreviewGrid: toRef(props, 'showPreviewGrid'),
})
</script>
