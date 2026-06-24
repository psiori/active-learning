<template>
  <label class="field">
    <span>Project</span>
    <select :value="selectedProject" :disabled="loadingConfig || !projects.length" @change="handleChange">
      <option v-for="project in projects" :key="project.project_name" :value="project.project_name">
        {{ project.project_name }}
      </option>
    </select>
  </label>
</template>

<script setup>
const props = defineProps({
  projects: {
    type: Array,
    required: true,
  },
  selectedProject: {
    type: String,
    required: true,
  },
  loadingConfig: {
    type: Boolean,
    required: true,
  },
})

const emit = defineEmits(['change'])

function handleChange(event) {
  emit('change', event.target.value)
}
</script>
