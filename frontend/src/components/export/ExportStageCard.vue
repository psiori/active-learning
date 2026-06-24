<template>
  <section class="stage-card">
    <div class="stage-header">
      <div class="stage-header__content">
        <p class="stage-kicker">Stage 3</p>
        <h2>Export</h2>
      </div>
      <button
        class="save-config-btn"
        :class="{
          'save-config-btn--success': saveState === 'success',
          'save-config-btn--error': saveState === 'error',
        }"
        :disabled="!isDirty || saveState === 'saving'"
        :title="isDirty ? 'Save export settings to config file' : 'No unsaved changes'"
        @click="handleSave"
      >
        <svg v-if="saveState === 'saving'" class="save-config-btn__spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
          <circle cx="12" cy="12" r="10" stroke-opacity="0.25" />
          <path d="M12 2a10 10 0 010 20" />
        </svg>
        <svg v-else-if="saveState === 'success'" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
        <svg v-else-if="saveState === 'error'" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
          <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
        </svg>
        <svg v-else width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z" />
          <polyline points="17 21 17 13 7 13 7 21" />
          <polyline points="7 3 7 8 15 8" />
        </svg>
      </button>
    </div>

    <ExportStagePanel
      :export-form="exportForm"
      :active-result="activeResult"
      :selected-strategy-ids="selectedStrategyIds"
      :sama-project-id="samaProjectId"
      :export-prefix="exportPrefix"
      :loading-export-overlay="loadingExportOverlay"
      :overlay-export-completed-at="overlayExportCompletedAt"
      :loading-export-seed="loadingExportSeed"
      :seed-confirmation="seedConfirmation"
      :disabled-input="disabledInput"
      :disable-overlay="disableOverlay"
      :disable-seed="disableSeed"
      @overlay="$emit('overlay')"
      @seed="$emit('seed')"
    />
  </section>
</template>

<script setup>
import { onBeforeUnmount, ref } from 'vue'
import ExportStagePanel from './ExportStagePanel.vue'
import { useActiveLearningStore } from '@/stores/activeLearningStore'

defineProps({
  exportForm: { type: Object, required: true },
  activeResult: { type: Object, default: null },
  selectedStrategyIds: { type: Array, required: true },
  samaProjectId: { type: Number, default: null },
  exportPrefix: { type: String, default: '' },
  loadingExportOverlay: { type: Boolean, required: true },
  overlayExportCompletedAt: { type: Number, required: true },
  loadingExportSeed: { type: Boolean, required: true },
  seedConfirmation: { type: Object, default: null },
  disabledInput: { type: Boolean, default: false },
  disableOverlay: { type: Boolean, required: true },
  disableSeed: { type: Boolean, required: true },
  isDirty: { type: Boolean, default: false },
})

defineEmits(['overlay', 'seed'])

const store = useActiveLearningStore()
const saveState = ref('idle')
let resetTimer = null

async function handleSave() {
  if (saveState.value !== 'idle') return
  saveState.value = 'saving'
  clearTimeout(resetTimer)
  try {
    await store.saveExportConfig()
    saveState.value = 'success'
  } catch (err) {
    console.error('Failed to save export config:', err)
    saveState.value = 'error'
  }
  resetTimer = setTimeout(() => { saveState.value = 'idle' }, 1500)
}

onBeforeUnmount(() => clearTimeout(resetTimer))
</script>

<style scoped>
.stage-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 4px;
}
</style>
