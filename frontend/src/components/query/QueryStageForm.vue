<template>
  <form class="query-form" @submit.prevent="emit('submit')">
    <label class="field checkbox-field">
      <input
        v-model="queryForm.brightness_filter_enabled"
        type="checkbox"
        :disabled="disabled"
      />
      <span>Apply brightness filter</span>
    </label>

    <label class="field">
      <span>Brightness range</span>
      <RangeSlider
        :min="BRIGHTNESS_MIN"
        :max="BRIGHTNESS_MAX"
        :step="1"
        :min-value="queryForm.min_brightness"
        :max-value="queryForm.max_brightness"
        :disabled="disabled || !queryForm.brightness_filter_enabled"
        min-aria-label="Minimum brightness"
        max-aria-label="Maximum brightness"
        @update:min-value="queryForm.min_brightness = $event"
        @update:max-value="queryForm.max_brightness = $event"
      />
    </label>

    <label class="field">
      <span>Date range</span>
      <DateRangeField
        :start-value="queryForm.start"
        :end-value="queryForm.end"
        :disabled="disabled"
        start-aria-label="Start date"
        end-aria-label="End date"
        @update:start-value="queryForm.start = $event"
        @update:end-value="queryForm.end = $event"
      />
      <span v-if="dateRangeError" class="field-error">{{ dateRangeError }}</span>
    </label>

    <label class="field">
      <span>Minimum duration between images</span>
      <div class="field-input-with-unit">
        <input
          v-model.number="queryForm.min_milliseconds_between_images"
          type="number"
          min="0"
          step="1"
          :disabled="disabled"
          aria-label="Minimum duration between images"
        />
        <span class="field-input-with-unit__suffix" aria-hidden="true">ms</span>
      </div>
    </label>

    <label class="field checkbox-field">
      <input v-model="queryForm.exclude_seeded" type="checkbox" :disabled="disabled" />
      <span>Exclude seeded</span>
    </label>

    <label class="field checkbox-field">
      <input v-model="queryForm.exclude_al_excluded" type="checkbox" :disabled="disabled" />
      <span>Exclude 'al-excluded' images</span>
    </label>

    <label class="field checkbox-field">
      <input v-model="queryForm.use_full_res_images" type="checkbox" :disabled="disabled" />
      <span>Use full resolution images</span>
    </label>

    <button
      v-if="cancellingQuery"
      class="primary-button"
      type="button"
      disabled
    >
      Cancelling…
    </button>
    <button
      v-else-if="loadingQuery"
      class="primary-button"
      type="button"
      @click="emit('cancel')"
    >
      Cancel query
    </button>
    <button v-else class="primary-button" type="submit" :disabled="disabled || !!dateRangeError">
      Run query
    </button>
  </form>
</template>

<script setup>
import { computed } from 'vue'
import DateRangeField from '@/components/base/DateRangeField.vue'
import RangeSlider from '@/components/base/RangeSlider.vue'

const BRIGHTNESS_MIN = 0
const BRIGHTNESS_MAX = 255

const props = defineProps({
  queryForm: {
    type: Object,
    required: true,
  },
  loadingQuery: {
    type: Boolean,
    required: true,
  },
  disabled: {
    type: Boolean,
    required: true,
  },
  cancellingQuery: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits(['submit', 'cancel'])

const dateRangeError = computed(() => {
  const { start, end } = props.queryForm
  if (start && end && start > end) return 'Start date must be before end date'
  return null
})
</script>
