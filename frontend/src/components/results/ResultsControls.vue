<template>
  <section class="results-controls">
    <div v-if="showOverlayControls" class="results-controls__section results-controls__section--overlay">
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

    <div v-if="showSelectionState" class="results-controls__divider" aria-hidden="true"></div>

    <div v-if="showSelectionState" class="results-controls__section results-controls__section--actions">
      <div class="results-controls__heading">
        <p class="results-controls__label">Selection actions</p>
        <p class="results-controls__meta">{{ excludedCount }} excluded</p>
      </div>
      <div class="results-controls__actions">
        <button
          class="btn-secondary"
          :class="{ 'is-disabled': toggleAllDisabled }"
          type="button"
          :disabled="toggleAllDisabled"
          :aria-disabled="toggleAllDisabled"
          @click="handleToggleAll"
        >
          {{ excludedCount === 0 ? 'Exclude all' : 'Include all' }}
        </button>
        <button
          class="btn-secondary btn-export-tags"
          :class="{
            'is-disabled': exportTagsDisabled,
            'btn-export-tags--persisted': showPersistedState,
          }"
          type="button"
          :disabled="exportTagsDisabled"
          :aria-disabled="exportTagsDisabled"
          @click="emit('exportTags')"
        >
          <span
            class="btn-export-tags__label"
            :class="{ 'btn-export-tags__label--pulse': loadingExportTags || showPersistedState }"
          >
            {{ buttonLabel }}
          </span>
        </button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import RangeSlider from '@/components/base/RangeSlider.vue'

const props = defineProps({
  overlayAlpha: {
    type: Number,
    required: true,
  },
  showOverlayControls: {
    type: Boolean,
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
})

const emit = defineEmits(['update:overlayAlpha', 'exportTags', 'includeAll', 'excludeAll'])
const toggleAllDisabled = computed(() => props.totalCount === 0)
const exportTagsDisabled = computed(() => props.loadingExportTags || props.excludedCount === 0)
const showPersistedState = ref(false)
let persistedTimer = null

const buttonLabel = computed(() => {
  if (props.loadingExportTags) return 'Persisting…'
  if (showPersistedState.value) return 'Persisted'
  return 'Persist exclusions in CRID'
})

function clearPersistedTimer() {
  if (persistedTimer) {
    window.clearTimeout(persistedTimer)
    persistedTimer = null
  }
}

watch(
  () => props.exclusionTagsPersistedAt,
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

function handleToggleAll() {
  if (props.excludedCount === 0) {
    emit('excludeAll')
    return
  }
  emit('includeAll')
}

function formatOpacityValue(value) {
  return `${value}%`
}

onBeforeUnmount(() => {
  clearPersistedTimer()
})
</script>
