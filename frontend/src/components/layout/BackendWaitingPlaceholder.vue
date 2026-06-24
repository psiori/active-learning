<template>
  <section class="backend-waiting-shell">
    <div class="backend-waiting-panel">
      <p class="stage-kicker">CRID</p>
      <h1>Active Learning</h1>
      <p class="backend-waiting-text">{{ statusText }}</p>
      <div class="backend-waiting-bar" aria-hidden="true">
        <span class="backend-waiting-bar__fill"></span>
      </div>
    </div>
  </section>
</template>

<script setup>
defineProps({
  statusText: {
    type: String,
    required: true,
  },
})
</script>
