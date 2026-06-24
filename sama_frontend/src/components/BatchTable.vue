<template>
  <section class="sama-card batch-card">
    <header class="batch-header">
      <div>
        <h2>Recent batches</h2>
        <p class="hint">
          {{ batches.length }} batch(es)
          {{ selectedCount > 0 ? `· ${selectedCount} selected` : '' }}
        </p>
      </div>
      <div class="batch-controls">
        <label class="control">
          Days
          <input
            type="number"
            :value="days"
            min="1"
            max="365"
            @change="$emit('update:days', Number($event.target.value))"
          />
        </label>
        <label class="control">
          <input
            type="checkbox"
            :checked="enrich"
            @change="$emit('update:enrich', $event.target.checked)"
          />
          Enrich (imported/masks/approved)
        </label>
        <button class="primary" :disabled="loading" @click="$emit('refresh')">
          {{ loading ? 'Loading…' : 'Refresh' }}
        </button>
      </div>
    </header>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th><input type="checkbox" :checked="allSelected" @change="toggleAll" /></th>
            <th>Route to</th>
            <th>Project</th>
            <th>Batch</th>
            <th>Updated</th>
            <th>State</th>
            <th>Tasks</th>
            <th>Status</th>
            <th>Imported</th>
            <th>Masks</th>
            <th>Approved</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="batch in batches" :key="rowKey(batch)" :class="rowClass(batch)">
            <td>
              <input
                type="checkbox"
                :checked="isSelected(batch)"
                @change="toggle(batch)"
              />
            </td>
            <td>
              <select
                v-if="(batch.consumers || []).length > 1"
                :value="chosenOrigin(batch)"
                @change="onChosenOrigin(batch, $event.target.value)"
              >
                <option
                  v-for="c in batch.consumers"
                  :key="c.origin"
                  :value="c.origin"
                >{{ c.origin }}</option>
              </select>
              <span v-else>{{ chosenOrigin(batch) }}</span>
            </td>
            <td>{{ batch.project_type }}/{{ batch.environment }}</td>
            <td>{{ batch.batch_id }}</td>
            <td>{{ formatDate(batch.updated_at) }}</td>
            <td>{{ batch.state || '' }}</td>
            <td>{{ batch.total_tasks ?? '' }}</td>
            <td class="counts">{{ formatCounts(batch.status_counts) }}</td>
            <td>{{ batch.imported ? '✓' : '' }}</td>
            <td>{{ batch.masks_exist ? '✓' : '' }}</td>
            <td>{{ batch.approved ? '✓' : '' }}</td>
          </tr>
          <tr v-if="!batches.length && !loading">
            <td colspan="11" class="empty">No batches in range.</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  batches: { type: Array, default: () => [] },
  selected: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
  days: { type: Number, default: 90 },
  enrich: { type: Boolean, default: true },
})

const emit = defineEmits([
  'update:selected',
  'update:days',
  'update:enrich',
  'update:chosen-origin',
  'refresh',
])

const selectedCount = computed(() => props.selected.length)
const allSelected = computed(
  () => props.batches.length > 0 && props.selected.length === props.batches.length,
)

function rowKey(batch) {
  return `${batch.project_id}-${batch.batch_id}`
}

function chosenOrigin(batch) {
  return batch.chosen_origin || batch.origin
}

function onChosenOrigin(batch, origin) {
  emit('update:chosen-origin', { batch, origin })
}

function isSelected(batch) {
  return props.selected.some((s) => rowKey(s) === rowKey(batch))
}

function toggle(batch) {
  const key = rowKey(batch)
  const next = isSelected(batch)
    ? props.selected.filter((s) => rowKey(s) !== key)
    : [...props.selected, batch]
  emit('update:selected', next)
}

function toggleAll() {
  if (allSelected.value) emit('update:selected', [])
  else emit('update:selected', [...props.batches])
}

function formatDate(value) {
  if (!value) return ''
  return value.replace('T', ' ').replace('Z', '')
}

function formatCounts(counts) {
  if (!counts) return ''
  return Object.entries(counts)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => `${k}=${v}`)
    .join(' ')
}

function rowClass(batch) {
  if (batch.approved) return 'approved'
  if (batch.masks_exist) return 'masks-ready'
  if (batch.imported) return 'imported'
  return ''
}
</script>

<style scoped>
.batch-card {
  background: var(--surface-card, #fff);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
.batch-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.75rem;
}
.batch-header h2 {
  margin: 0;
  font-size: 1rem;
}
.hint {
  font-size: 0.8rem;
  color: #666;
  margin: 0.25rem 0 0;
}
.batch-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}
.control {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.85rem;
}
.control input[type='number'] {
  width: 60px;
}
.primary {
  background: #3366cc;
  color: #fff;
  border: none;
  padding: 0.4rem 0.9rem;
  border-radius: 6px;
  cursor: pointer;
}
.primary:disabled {
  background: #aaa;
  cursor: not-allowed;
}
.table-wrap {
  overflow-x: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
th,
td {
  text-align: left;
  padding: 0.35rem 0.5rem;
  border-bottom: 1px solid #eee;
  white-space: nowrap;
}
th {
  background: #fafafa;
  font-weight: 600;
}
.empty {
  color: #888;
  text-align: center;
  padding: 1rem;
}
.counts {
  font-family: monospace;
  font-size: 0.75rem;
  color: #444;
}
tr.imported {
  background: #fafff8;
}
tr.masks-ready {
  background: #fff8e6;
}
tr.approved {
  background: #e8f5e9;
}
</style>
