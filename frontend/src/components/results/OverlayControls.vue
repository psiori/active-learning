<template>
  <section class="overlay-controls">
    <label class="field">
      <span>Mask overlay strength</span>
      <input :value="overlayAlpha" type="range" min="0" max="100" step="1" @input="handleInput" />
    </label>
  </section>
</template>

<script setup>
defineProps({
  overlayAlpha: {
    type: Number,
    required: true,
  },
})

const emit = defineEmits(['update:overlayAlpha'])

function handleInput(event) {
  emit('update:overlayAlpha', Number(event.target.value))
}
</script>
