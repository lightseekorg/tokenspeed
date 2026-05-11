import { defineConfig } from "vitepress";

export default defineConfig({
  lang: "en-US",
  title: "TokenSpeed",
  description: "TokenSpeed is a speed-of-light LLM inference engine.",
  base: "/tokenspeed/",
  srcDir: ".",
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ["meta", { property: "og:title", content: "TokenSpeed Docs" }],
    [
      "meta",
      {
        property: "og:description",
        content:
          "Guides and reference for running TokenSpeed, configuring the server, and operating multi-GPU inference."
      }
    ]
  ],
  themeConfig: {
    nav: [
      { text: "Docs Home", link: "/" },
      { text: "Getting Started", link: "/guides/getting-started" },
      { text: "Launching", link: "/guides/launching" },
      { text: "Recipes", link: "/recipes/models" },
      { text: "Configuration", link: "/configuration/server" },
      { text: "GitHub", link: "https://github.com/lightseekorg/tokenspeed" }
    ],
    sidebar: [
      {
        text: "Overview",
        items: [{ text: "Docs Home", link: "/" }]
      },
      {
        text: "Guides",
        items: [
          { text: "Getting Started", link: "/guides/getting-started" },
          { text: "Launching a Server", link: "/guides/launching" }
        ]
      },
      {
        text: "Configuration",
        items: [
          { text: "Server Parameters", link: "/configuration/server" },
          {
            text: "Compatible Parameters",
            link: "/configuration/compatible-parameters"
          }
        ]
      },
      {
        text: "Recipes",
        items: [{ text: "Model Recipes", link: "/recipes/models" }]
      },
      {
        text: "Serving",
        items: [
          { text: "Scheduler", link: "/serving/scheduler" },
          { text: "Parallelism", link: "/serving/parallelism" }
        ]
      }
    ],
    search: {
      provider: "local"
    },
    editLink: {
      pattern:
        "https://github.com/lightseekorg/tokenspeed/edit/main/docs/:path",
      text: "Edit this page on GitHub"
    },
    footer: {
      message: "TokenSpeed documentation",
      copyright: "Copyright © lightseekorg"
    },
    socialLinks: [
      { icon: "github", link: "https://github.com/lightseekorg/tokenspeed" }
    ],
    outline: "deep",
    outlineTitle: "On this page",
    docFooter: {
      prev: "Previous page",
      next: "Next page"
    }
  }
});
