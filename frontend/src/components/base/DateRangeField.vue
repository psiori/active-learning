<template>
  <div class="date-range-field" :class="{ 'is-disabled': disabled }">
    <input
      :value="normalizedStartValue"
      type="date"
      :max="normalizedEndValue || undefined"
      :disabled="disabled"
      :aria-label="startAriaLabel"
      @input="emit('update:startValue', normalizeDateValue($event.target.value))"
    />
    <div class="date-range-field__divider" aria-hidden="true">–</div>
    <input
      :value="normalizedEndValue"
      type="date"
      :min="normalizedStartValue || undefined"
      :disabled="disabled"
      :aria-label="endAriaLabel"
      @input="emit('update:endValue', normalizeDateValue($event.target.value))"
    />
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  startValue: {
    type: String,
    default: '',
  },
  endValue: {
    type: String,
    default: '',
  },
  disabled: {
    type: Boolean,
    default: false,
  },
  startAriaLabel: {
    type: String,
    default: 'Start date',
  },
  endAriaLabel: {
    type: String,
    default: 'End date',
  },
})

const emit = defineEmits(['update:startValue', 'update:endValue'])

const normalizedStartValue = computed(() => normalizeDateValue(props.startValue))
const normalizedEndValue = computed(() => normalizeDateValue(props.endValue))

function normalizeDateValue(value) {
  if (!value) {
    return ''
  }
  return /^\d{4}-\d{2}-\d{2}$/.test(value) ? value : ''
}
</script>

<style scoped>
.date-range-field {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 8px;
  align-items: center;
}

.date-range-field.is-disabled {
  opacity: 0.65;
}

.date-range-field input {
  width: 100%;
  min-width: 0;
  padding-inline: 6px;
  font-size: 0.9rem;
}

.date-range-field__divider {
  color: #5c677d;
  font-size: 0.9rem;
  text-align: center;
}

@media (max-width: 720px) {
  .date-range-field {
    grid-template-columns: 1fr auto 1fr;
  }
}
</style>
