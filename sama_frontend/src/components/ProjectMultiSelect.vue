<template>
  <section class="sama-card">
    <header>
      <h2>Projects</h2>
      <p class="hint">Pick origins to include in batch listing + pipeline runs.</p>
    </header>
    <div v-if="loading" class="loading">Loading projects…</div>
    <ul v-else class="origin-list">
      <li v-for="origin in origins" :key="origin">
        <label>
          <input
            type="checkbox"
            :checked="selected.includes(origin)"
            @change="toggle(origin)"
          />
          <span class="origin-name">{{ origin }}</span>
          <span class="origin-meta">{{ summary(origin) }}</span>
        </label>
      </li>
    </ul>
    <div class="actions">
      <button class="link" @click="emit('update:selected', [...origins])">All</button>
      <button class="link" @click="emit('update:selected', [])">None</button>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  projects: { type: Array, default: () => [] },
  selected: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
})
const emit = defineEmits(['update:selected'])

const origins = computed(() => {
  const set = new Set()
  for (const p of props.projects) set.add(p.origin)
  return [...set].sort()
})

function summary(origin) {
  const entries = props.projects.filter((p) => p.origin === origin)
  const cams = new Set(entries.map((e) => e.project_type))
  return [...cams].join(', ')
}

function toggle(origin) {
  const next = props.selected.includes(origin)
    ? props.selected.filter((o) => o !== origin)
    : [...props.selected, origin]
  emit('update:selected', next)
}
</script>

<style scoped>
.sama-card {
  background: var(--surface-card, #fff);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
.sama-card header h2 {
  margin: 0;
  font-size: 1rem;
}
.hint {
  font-size: 0.8rem;
  color: #666;
  margin: 0.25rem 0 0.75rem;
}
.origin-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}
.origin-list label {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  cursor: pointer;
}
.origin-name {
  font-weight: 600;
}
.origin-meta {
  color: #666;
  font-size: 0.8rem;
}
.actions {
  margin-top: 0.75rem;
  display: flex;
  gap: 0.75rem;
}
.link {
  background: none;
  border: none;
  color: #3366cc;
  cursor: pointer;
  font-size: 0.85rem;
  padding: 0;
}
.loading {
  color: #888;
  font-size: 0.9rem;
}
</style>
