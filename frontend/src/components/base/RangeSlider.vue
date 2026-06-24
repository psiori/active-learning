<template>
  <div class="range-slider" :class="{ 'is-disabled': disabled }">
    <div class="range-slider__slider">
      <div class="range-slider__track"></div>
      <div class="range-slider__active" :style="activeTrackStyle"></div>
      <input
        v-if="!singleValue"
        class="range-slider__input range-slider__input--min"
        :value="clampedMinValue"
        type="range"
        :min="min"
        :max="max"
        :step="step"
        :disabled="disabled"
        :style="{ zIndex: minHandleZIndex }"
        :aria-label="minAriaLabel"
        @input="handleMinInput"
        @focus="focusedInput = 'min'"
        @blur="focusedInput = null"
      />
      <input
        class="range-slider__input range-slider__input--max"
        :value="clampedMaxValue"
        type="range"
        :min="min"
        :max="max"
        :step="step"
        :disabled="disabled"
        :style="{ zIndex: maxHandleZIndex }"
        :aria-label="maxAriaLabel"
        @input="handleMaxInput"
        @focus="focusedInput = 'max'"
        @blur="focusedInput = null"
      />
      <div
        v-if="!singleValue"
        class="range-slider__thumb-value"
        :class="{ 'is-focused': focusedInput === 'min' }"
        :style="minThumbValueStyle"
        aria-hidden="true"
      >{{ formattedMinValue }}</div>
      <div
        class="range-slider__thumb-value"
        :class="{ 'is-focused': focusedInput === 'max' }"
        :style="maxThumbValueStyle"
        aria-hidden="true"
      >{{ formattedMaxValue }}</div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

const props = defineProps({
  min: {
    type: Number,
    required: true,
  },
  max: {
    type: Number,
    required: true,
  },
  step: {
    type: Number,
    default: 1,
  },
  minValue: {
    type: Number,
    required: true,
  },
  maxValue: {
    type: Number,
    required: true,
  },
  disabled: {
    type: Boolean,
    default: false,
  },
  minAriaLabel: {
    type: String,
    default: 'Minimum value',
  },
  maxAriaLabel: {
    type: String,
    default: 'Maximum value',
  },
  singleValue: {
    type: Boolean,
    default: false,
  },
  formatValue: {
    type: Function,
    default: null,
  },
})

const emit = defineEmits(['update:minValue', 'update:maxValue'])

const currentMinValue = ref(props.min)
const currentMaxValue = ref(props.max)
const focusedInput = ref(null)

const clampedMinValue = computed(() => currentMinValue.value)
const clampedMaxValue = computed(() => currentMaxValue.value)
const formattedMinValue = computed(() => formatSummaryValue(clampedMinValue.value))
const formattedMaxValue = computed(() => formatSummaryValue(clampedMaxValue.value))

function thumbValueStyle(percent) {
  // Corrects for browsers placing thumb flush with track edges at 0% and 100%
  return { left: `calc(${percent}% + ${(0.5 - percent / 100) * 20}px)` }
}

const minThumbValueStyle = computed(() => thumbValueStyle(valueToPercent(clampedMinValue.value)))
const maxThumbValueStyle = computed(() => thumbValueStyle(valueToPercent(clampedMaxValue.value)))

const activeTrackStyle = computed(() => {
  const minPercent = props.singleValue ? 0 : valueToPercent(clampedMinValue.value)
  const maxPercent = valueToPercent(clampedMaxValue.value)
  return {
    left: `${minPercent}%`,
    width: `${Math.max(maxPercent - minPercent, 0)}%`,
  }
})

const minHandleZIndex = computed(() => (
  clampedMinValue.value >= clampedMaxValue.value - props.step ? 4 : 2
))

const maxHandleZIndex = computed(() => (
  clampedMaxValue.value <= clampedMinValue.value + props.step ? 5 : 3
))

watch(
  () => [props.minValue, props.maxValue],
  ([nextMinValue, nextMaxValue]) => {
    const normalizedMin = props.singleValue ? props.min : clampValue(nextMinValue)
    const normalizedMax = clampValue(nextMaxValue)

    currentMinValue.value = Math.min(normalizedMin, normalizedMax)
    currentMaxValue.value = Math.max(normalizedMax, currentMinValue.value)
  },
  { immediate: true },
)

watch(
  () => currentMinValue.value,
  (nextValue) => {
    if (nextValue !== props.minValue) {
      emit('update:minValue', nextValue)
    }
  },
)

watch(
  () => currentMaxValue.value,
  (nextValue) => {
    if (nextValue !== props.maxValue) {
      emit('update:maxValue', nextValue)
    }
  },
)

function clampValue(value) {
  const numeric = Number(value)
  if (Number.isNaN(numeric)) {
    return props.min
  }
  return Math.min(Math.max(numeric, props.min), props.max)
}

function valueToPercent(value) {
  if (props.max === props.min) {
    return 0
  }
  return ((value - props.min) / (props.max - props.min)) * 100
}

function formatSummaryValue(value) {
  if (typeof props.formatValue === 'function') {
    return props.formatValue(value)
  }
  return String(value)
}

function handleMinInput(event) {
  const nextValue = clampValue(event.target.value)
  const boundedValue = Math.min(nextValue, currentMaxValue.value)
  currentMinValue.value = boundedValue
  event.target.value = String(boundedValue)
}

function handleMaxInput(event) {
  const nextValue = clampValue(event.target.value)
  const boundedValue = Math.max(nextValue, currentMinValue.value)
  currentMaxValue.value = boundedValue
  event.target.value = String(boundedValue)
}
</script>

<style scoped>
.range-slider {
  display: flex;
  flex-direction: column;
}

.range-slider.is-disabled {
  opacity: 0.65;
}

.range-slider__slider {
  position: relative;
  height: 32px;
}

.range-slider__track,
.range-slider__active {
  position: absolute;
  top: 50%;
  height: 8px;
  border-radius: 999px;
  transform: translateY(-50%);
}

.range-slider__track {
  left: 0;
  right: 0;
  background: rgba(20, 33, 61, 0.12);
  box-shadow: inset 0 1px 2px rgba(20, 33, 61, 0.08);
}

.range-slider__active {
  background: linear-gradient(135deg, #264653 0%, #2a9d8f 100%);
  box-shadow: 0 0 0 1px rgba(38, 70, 83, 0.12), 0 2px 8px rgba(38, 70, 83, 0.18);
}

.range-slider__input {
  position: absolute;
  left: 0;
  width: 100%;
  height: 32px;
  top: 0;
  margin: 0;
  padding: 0;
  appearance: none;
  pointer-events: none;
  border: 0;
  background: transparent;
}

.range-slider__input--min {
  z-index: 2;
}

.range-slider__input--max {
  z-index: 3;
}

.range-slider__input::-webkit-slider-runnable-track {
  height: 8px;
  background: transparent;
}

.range-slider__input::-moz-range-track {
  height: 8px;
  background: transparent;
}

.range-slider__input::-webkit-slider-thumb {
  appearance: none;
  width: 20px;
  height: 20px;
  margin-top: -6px;
  border: none;
  border-radius: 50%;
  background: transparent;
  cursor: pointer;
  pointer-events: auto;
}

.range-slider__input:focus {
  outline: none;
}

.range-slider__input::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border: none;
  border-radius: 50%;
  background: transparent;
  cursor: pointer;
  pointer-events: auto;
}

.range-slider__thumb-value {
  position: absolute;
  top: 50%;
  transform: translateX(-50%) translateY(-50%);
  height: 24px;
  min-width: 24px;
  padding: 0 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #264653 0%, #2a9d8f 100%);
  border: 2px solid #fff;
  border-radius: 999px;
  color: #fff;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.02em;
  box-shadow: 0 3px 10px rgba(20, 33, 61, 0.24);
  pointer-events: none;
  white-space: nowrap;
  user-select: none;
  z-index: 10;
}

.range-slider__thumb-value.is-focused {
  box-shadow: 0 0 0 4px rgba(42, 157, 143, 0.2), 0 3px 10px rgba(20, 33, 61, 0.24);
}
</style>
