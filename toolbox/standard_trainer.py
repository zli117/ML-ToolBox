from toolbox.trainer_base import BaseTrainer


class StandardTrainer(BaseTrainer):

    def train_one_epoch(self) -> None:
        if self._optimizer is not None:
            self.model.train()
            total_steps = len(self.train_loader)
            for batch in self.train_loader.next_epoch():
                ipt, target = self.parse_train_batch(batch)
                self.trigger_call_backs('on_train_batch_begin',
                                        curr_step=self._curr_step,
                                        total_steps=total_steps,
                                        batch=(ipt, target))
                output = self.model(ipt)
                loss = self.loss_fn(output, target)
                self._optimizer.zero_grad()
                loss.backward()
                # TODO: Add support for closure
                self._optimizer.step()
                self.trigger_call_backs('on_train_batch_end',
                                        curr_step=self._curr_step,
                                        total_steps=total_steps, loss=loss)
                self._curr_step += 1
            self._curr_step = 0
