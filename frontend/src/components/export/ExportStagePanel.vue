<template>
  <div>
<p v-if="activeResult?.result_kind === 'strategy'" class="stage-note">
      {{ selectedStrategyIds.length }} of {{ activeResult.selected_ids_count }} currently selected
    </p>
    <label class="field">
      <span>Sama project id</span>
      <input :value="samaProjectId ?? 'n/a'" type="text" readonly disabled />
    </label>
    <label class="field">
      <span>Export prefix</span>
      <input :value="exportPrefix || 'n/a'" type="text" readonly disabled />
    </label>
    <label class="field">
      <span>Sama export priority</span>
      <input
        v-model.number="exportForm.sama_priority"
        type="number"
        step="1"
        :disabled="disabledInput"
      />
    </label>
    <p v-if="samaProjectId == null" class="stage-note stage-note--warn">
      No <code>export.sama_project_id</code> in project config: export only registers a CRID dataset (no Sama batch). The priority field is kept for future Sama submissions and can still be saved to config.
    </p>
    <div class="query-form">
      <button
        class="secondary-button export-button"
        :class="{ 'export-button--persisted': showPersistedState }"
        type="button"
        :disabled="disableOverlay"
        @click="emit('overlay')"
      >
        <span class="export-button__label" :class="{ 'export-button__label--pulse': loadingExportOverlay || showPersistedState }">
          {{ overlayButtonLabel }}
        </span>
      </button>
      <button
        class="primary-button"
        type="button"
        :disabled="disableSeed"
        @click="emit('seed')"
      >
        {{ seedButtonLabel }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'

const props = defineProps({
  exportForm: {
    type: Object,
    required: true,
  },
  samaProjectId: {
    type: Number,
    default: null,
  },
  exportPrefix: {
    type: String,
    default: '',
  },
  activeResult: {
    type: Object,
    default: null,
  },
  selectedStrategyIds: {
    type: Array,
    required: true,
  },
  loadingExportOverlay: {
    type: Boolean,
    required: true,
  },
  overlayExportCompletedAt: {
    type: Number,
    required: true,
  },
  loadingExportSeed: {
    type: Boolean,
    required: true,
  },
  seedConfirmation: {
    type: Object,
    default: null,
  },
  disabledInput: {
    type: Boolean,
    default: false,
  },
  disableOverlay: {
    type: Boolean,
    required: true,
  },
  disableSeed: {
    type: Boolean,
    required: true,
  },
})

const emit = defineEmits(['overlay', 'seed'])
const showPersistedState = ref(false)
let persistedTimer = null

const overlayButtonLabel = computed(() => {
  if (props.loadingExportOverlay) return 'Preparing overlay mosaic...'
  if (showPersistedState.value) return 'Persisted'
  return 'Download overlay mosaic'
})

function clearPersistedTimer() {
  if (persistedTimer) {
    window.clearTimeout(persistedTimer)
    persistedTimer = null
  }
}

watch(
  () => props.overlayExportCompletedAt,
  (next, previous) => {
    if (!next || next === previous) {
      return
    }

    clearPersistedTimer()
    showPersistedState.value = true
    persistedTimer = window.setTimeout(() => {
      showPersistedState.value = false
      persistedTimer = null
    }, 2600)
  },
)

const seedButtonLabel = computed(() => {
  if (props.loadingExportSeed) {
    return props.seedConfirmation ? 'Confirming seed...' : (props.samaProjectId != null ? 'Seeding to Sama...' : 'Exporting to CRID...')
  }
  if (props.seedConfirmation) return 'Confirm seed'
  return props.samaProjectId != null ? 'Seed to Sama' : 'Export to CRID'
})

onBeforeUnmount(() => {
  clearPersistedTimer()
})
</script>
