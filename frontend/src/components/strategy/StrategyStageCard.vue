<template>
  <section class="stage-card">
    <div class="stage-header">
      <div class="stage-header__content">
        <p class="stage-kicker">Stage 2</p>
        <h2>Strategy</h2>
      </div>
      <button
        class="save-config-btn"
        :class="{
          'save-config-btn--success': saveState === 'success',
          'save-config-btn--error': saveState === 'error',
        }"
        :disabled="!isDirty || saveState === 'saving'"
        :title="isDirty ? 'Save strategy settings to config file' : 'No unsaved changes'"
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
    <StrategyStageForm
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
      :disabled-select="disabledSelect"
      :disabled-submit="disabledSubmit"
      :cancelling-strategy="cancellingStrategy"
      @submit="$emit('submit')"
      @cancel="$emit('cancel')"
      @sync-model-urls="$emit('sync-model-urls')"
    />
  </section>
</template>

<script setup>
import { ref, onBeforeUnmount } from 'vue'
import StrategyStageForm from './StrategyStageForm.vue'
import { useActiveLearningStore } from '@/stores/activeLearningStore'

defineProps({
  selectionForm: { type: Object, required: true },
  strategyParams: { type: Object, required: true },
  availableStrategies: { type: Array, default: () => [] },
  featureModels: { type: Array, default: () => [] },
  uncertaintyModels: { type: Array, default: () => [] },
  usesCoresetParams: { type: Boolean, default: false },
  usesUncertaintyParams: { type: Boolean, default: false },
  usesAlgesParams: { type: Boolean, default: false },
  selectedUncertaintyModel: { type: Object, default: null },
  selectedAlgesModel: { type: Object, default: null },
  loadingStrategy: { type: Boolean, default: false },
  disabledSelect: { type: Boolean, default: false },
  disabledSubmit: { type: Boolean, default: false },
  cancellingStrategy: { type: Boolean, default: false },
  isDirty: { type: Boolean, default: false },
})

defineEmits(['submit', 'cancel', 'sync-model-urls'])

const store = useActiveLearningStore()
const saveState = ref('idle')
let resetTimer = null

async function handleSave() {
  if (saveState.value !== 'idle') return
  saveState.value = 'saving'
  clearTimeout(resetTimer)
  try {
    await store.saveStrategyConfig()
    saveState.value = 'success'
  } catch (err) {
    console.error('Failed to save strategy config:', err)
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
