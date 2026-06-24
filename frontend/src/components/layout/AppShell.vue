<template>
  <div class="app-shell">
    <aside class="sidebar">
      <slot name="sidebar" />
    </aside>
    <main class="results-pane">
      <slot name="main" />
    </main>
  </div>
</template>
