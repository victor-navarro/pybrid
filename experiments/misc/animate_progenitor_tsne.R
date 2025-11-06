# Write a function that can be called from
# python to animate t-SNE data across epochs

# An R function to animate t-SNE data across epochs
library(tidyverse)
library(animation)



# Define function
animate_tsne <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  data_path <- args[1]
  filename <- args[2]
  save_path <- args[3]

  # data_path <- "results/exp_1/tsne/0/animations/progenitor/tsne_data_hybrid_layer_0.csv"
  # save_path <- "results/exp_1/tsne/0/animations/progenitor"
  # filename <- "progenitor_tsne_animation.gif"

  ppi <- 300
  width <- 4.5 * ppi
  height <- 3.5 * ppi

  # load data
  data <- read.csv(data_path)
  x_range <- range(data$tsne_1)
  y_range <- range(data$tsne_2)

  plot_epochs <- sort(unique(data$epoch))
  padding <- 10
  if (padding > 0) {
    plot_epochs <- c(rep(min(plot_epochs), padding), plot_epochs)
    plot_epochs <- c(plot_epochs, rep(max(plot_epochs), padding))
  }

  plotlist <- sapply(plot_epochs, function(e) {
    h <- data |>
      filter(epoch <= e) |>
      arrange(epoch)


    max_alpha <- max(h$epoch)
    trail_off <- 0.5

    h |>
      ggplot(aes(x = tsne_1, y = tsne_2, colour = class_names)) +
      geom_path(alpha = .2) +
      geom_point(aes(alpha = exp(trail_off * (epoch - max_alpha)))) +
      theme_bw() +
      ylim(y_range) +
      xlim(x_range) +
      guides(alpha = "none") +
      labs(
        x = "t-SNE 1", y = "t-SNE 2",
        colour = "Class", title = sprintf("Epoch: %d", e)
      )
  }, simplify = FALSE)

  setwd(save_path)
  saveGIF(
    {
      for (i in seq_along(plotlist)) {
        print(plotlist[[i]])
      }
    },
    movie.name = filename,
    ani.width = width,
    ani.height = height,
    interval = 0.5,
    ani.res = ppi
  )
  # also save the last plot as a png
  ggplot2::ggsave(
    gsub(".gif", ".png", filename),
    plotlist[[length(plotlist)]],
    width = width / ppi, height = height / ppi
  )
}



animate_tsne()
